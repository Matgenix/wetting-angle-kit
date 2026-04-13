import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .surface_definition import (
    HyperbolicTangentModel,
)


class ContactAngleBinning:
    """Binning-based contact angle estimator using density field fitting.

    Frames aggregated in spatial bins form a time-averaged density field.
    A hyperbolic tangent interface model is fitted and the implied contact
    angle is computed from fitted geometric parameters.
    """

    def __init__(
        self,
        parser,
        atom_indices,
        droplet_geometry="spherical",
        width_cylinder=None,
        binning_params=None,
        output_dir="output_analysis/",
        plot_graphs=True,
    ):
        self.parser = parser
        self.atom_indices = atom_indices
        self.droplet_geometry = droplet_geometry
        self.width_cylinder = width_cylinder
        self.output_dir = output_dir
        self.plot_graphs = plot_graphs
        if binning_params is None:
            max_dist = int(
                np.max(
                    np.array(
                        [
                            parser.box_size_y(frame_index=1),
                            parser.box_size_x(frame_index=1),
                        ]
                    )
                )
                / 3
            )
            self.binning_params = {
                "xi_0": 0,
                "xi_f": max_dist,
                "nbins_xi": 50,
                "zi_0": 0.0,
                "zi_f": max_dist,
                "nbins_zi": 50,
            }
        else:
            self.binning_params = binning_params
        self._initialize_grid()
        if self.width_cylinder is None:
            if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
                if self.droplet_geometry == "cylinder_x":
                    self.width_cylinder = self.parser.box_size_x(frame_index=1)
                elif self.droplet_geometry == "cylinder_y":
                    self.width_cylinder = self.parser.box_size_y(frame_index=1)
        os.makedirs(self.output_dir, exist_ok=True)
        matplotlib.use("Agg")

    def _initialize_grid(self):
        """Initialize bin edges, centers and cell sizes from parameters."""
        self.xi = np.linspace(
            self.binning_params["xi_0"],
            self.binning_params["xi_f"],
            self.binning_params["nbins_xi"],
        )
        self.zi = np.linspace(
            self.binning_params["zi_0"],
            self.binning_params["zi_f"],
            self.binning_params["nbins_zi"],
        )
        self.dxi = self.xi[1] - self.xi[0]
        self.dzi = self.zi[1] - self.zi[0]
        self.xi_cc = 0.5 * (self.xi[1:] + self.xi[:-1])
        self.zi_cc = 0.5 * (self.zi[1:] + self.zi[:-1])

    def binning(self, xi_par, zi_par, len_frames):
        """Return 2D density field by binning particle coordinates.

        Parameters
        ----------
        xi_par : ndarray
            Radial/in-plane coordinate values for particles over frames.
        zi_par : ndarray
            Vertical coordinate values for particles over frames.
        len_frames : int
            Number of frames aggregated.
        droplet_geometry : str, optional
            Override instance model type.
        width_cylinder : float, optional
            Override cylinder width.

        Returns
        -------
        ndarray, shape (nbins_xi-1, nbins_zi-1)
            Averaged density field on cell centers.
        """
        print(f"Binning with model: {self.droplet_geometry} ...")
        rho_cc = np.zeros((len(self.xi_cc), len(self.zi_cc)))
        xi_par_0, zi_par_0 = copy.deepcopy(xi_par), copy.deepcopy(zi_par)
        for i in range(len(self.xi_cc)):
            if i % 10 == 0:
                print(f"Advancement: {100 * i / (len(self.xi_cc) - 1):.2f}%")
            if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
                dV = 2 * self.width_cylinder * self.dxi * self.dzi
            elif self.droplet_geometry == "spherical":
                dV = 2 * np.pi * (self.xi_cc[i]) * self.dxi * self.dzi
            else:
                raise ValueError("Unknown model type: {}".format(self.droplet_geometry))
            for j in range(len(self.zi_cc)):
                where = (
                    (xi_par_0 > self.xi[i])
                    * (xi_par_0 < self.xi[i + 1])
                    * (zi_par_0 > self.zi[j])
                    * (zi_par_0 < self.zi[j + 1])
                )
                count_i = where.sum()
                rho_cc[i, j] = count_i / dV
                in_this_bin_indx = np.nonzero(where)
                xi_par_0 = np.delete(xi_par_0, in_this_bin_indx)
                zi_par_0 = np.delete(zi_par_0, in_this_bin_indx)
        if len_frames > 0:
            rho_cc /= len_frames
        return rho_cc

    def plot_density_with_isoline(
        self,
        xi_cc,
        zi_cc,
        rho_cc,
        circle_xi,
        circle_zi,
        wall_line_xi,
        wall_line_zi,
        batch_index=None,
        clevels=20,
        scale=0.75,
        close=True,
    ):
        """Plot density contour with fitted iso-surface approximations.

        Parameters
        ----------
        xi_cc, zi_cc : ndarray
            Cell center coordinates.
        rho_cc : ndarray
            Density field.
        circle_xi, circle_zi : ndarray
            Fitted circle isoline coordinates.
        wall_line_xi, wall_line_zi : ndarray
            Wall line coordinates.
        batch_index : int, optional
            Batch identifier for file naming.
        clevels : int, default 20
            Number of contour levels.
        scale : float, default 0.75
            Figure size scaling factor.
        close : bool, default True
            If True, close figure after saving.
        """
        name = (
            f"bin_plot_batch_{batch_index}.png"
            if batch_index is not None
            else "bin_plot.png"
        )
        plt.figure(dpi=300, figsize=(4 * scale, 3 * scale))
        plt.contourf(xi_cc, zi_cc, np.transpose(rho_cc), levels=clevels, cmap="jet")
        plt.colorbar()
        plt.plot(circle_xi, circle_zi, "--", color="black")
        plt.plot(wall_line_xi, wall_line_zi, "--", color="black")
        plt.savefig(os.path.join(self.output_dir, name))
        if close:
            plt.close()

    def save_logfile(
        self,
        particles_number,
        param_strings,
        theta,
        xi_cc,
        zi_cc,
        rho_cc,
        batch_index=None,
    ):
        """Write fitted parameters and density field CSV for a batch.

        Parameters
        ----------
        particles_number : float
            Average number of particles per frame in batch.
        param_strings : list[str]
            Formatted parameter lines from model.
        theta : float
            Contact angle in degrees.
        xi_cc, zi_cc : ndarray
            Cell centers.
        rho_cc : ndarray
            Density field.
        batch_index : int, optional
            Batch identifier for file naming.
        """
        batch_str = f"_batch_{batch_index}" if batch_index is not None else ""
        with open(os.path.join(self.output_dir, f"log_data{batch_str}.txt"), "w") as f:
            f.write("Simulation parameters:\n")
            f.write(f"reduced_particles_number:{particles_number}\n")
            f.write(f"model_type:{self.droplet_geometry}\n")
            if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
                f.write(f"width_cylinder:{self.width_cylinder}\n")
            f.write("Fitted parameters:\n")
            for param in param_strings:
                f.write(param)
            f.write(f"\n\nContact angle:{theta}")
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(zi_cc, xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_xi_cc = msh_xi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_rho_cc = rho_cc.reshape((len(xi_cc) * len(zi_cc)), order="F")
        CSV = np.c_[msh_xi_cc, msh_zi_cc, msh_rho_cc]
        np.savetxt(
            os.path.join(self.output_dir, f"rho_field{batch_str}.csv"),
            CSV,
            delimiter=",",
            header=(f"x_{len(xi_cc)},y_{len(zi_cc)}," f"rho_{len(xi_cc) * len(zi_cc)}"),
        )

    def process_batch(self, frame_list, model=None, batch_index=None):
        """Process a batch of frames and compute contact angle.

        Parameters
        ----------
        frame_list : sequence[int]
            Frame indices in the batch.
        model : SurfaceModel, optional
            Pre-existing fitted model instance; new model created if None.
        batch_index : int, optional
            Identifier appended to output filenames.

        Returns
        -------
        tuple(float, SurfaceModel)
            (contact_angle_degrees, fitted_model).
        """
        xi_par, zi_par, len_frames = self.parser.get_profile_coordinates(
            frame_indices=frame_list,
            droplet_geometry=self.droplet_geometry,
            atom_indices=self.atom_indices,
        )
        particles_number = len(xi_par) / max(len_frames, 1)
        print(
            f"\nNumber of fluid particles in batch"
            f"{(' ' + str(batch_index)) if batch_index is not None else ''}:"
            f"\t{particles_number}\n"
        )
        rho_cc = self.binning(xi_par, zi_par, len_frames)
        if model is None:
            model = HyperbolicTangentModel()
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(self.zi_cc, self.xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape(
            (len(self.xi_cc) * len(self.zi_cc)), order="F"
        )
        msh_xi_cc = msh_xi_cc_grid.reshape(
            (len(self.xi_cc) * len(self.zi_cc)), order="F"
        )
        msh_rho_cc = rho_cc.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        x_data = (msh_xi_cc, msh_zi_cc)
        model.fit(x_data, msh_rho_cc)
        param_strings = model.get_parameter_strings()
        print(
            "\nFitted parameters for batch{}:".format(
                f" {batch_index}" if batch_index is not None else ""
            )
        )
        print("".join(param_strings))
        contact_angle = model.compute_contact_angle()
        print(
            f"Contact angle for batch"
            f"{(' ' + str(batch_index)) if batch_index is not None else ''}:\t"
            f"{contact_angle}"
        )
        circle_xi, circle_zi, wall_line_xi, wall_line_zi = model.compute_isoline()
        if self.plot_graphs:
            self.plot_density_with_isoline(
                self.xi_cc,
                self.zi_cc,
                rho_cc,
                circle_xi,
                circle_zi,
                wall_line_xi,
                wall_line_zi,
                batch_index,
            )
        self.save_logfile(
            particles_number,
            param_strings,
            contact_angle,
            self.xi_cc,
            self.zi_cc,
            rho_cc,
            batch_index,
        )
        return contact_angle, model

    def process_all_batches(self, batch_size=100, save_angles=True):
        """Process all frames in batches returning list of contact angles.

        Parameters
        ----------
        batch_size : int, default 100
            Number of frames per batch.
        save_angles : bool, default True
            If True, save angle list as numpy file.

        Returns
        -------
        list[float]
            Contact angles per processed batch.
        """
        frames_tot = self.parser.frame_tot()
        print(f"Total frames: {frames_tot}")
        angles: list[float] = []
        for batch_index, start_frame in enumerate(range(0, frames_tot, batch_size)):
            frame_list = list(
                range(start_frame, min(start_frame + batch_size, frames_tot))
            )
            angle, _ = self.process_batch(frame_list, batch_index=batch_index + 1)
            angles.append(angle)
        if save_angles:
            np.save(
                os.path.join(
                    self.output_dir, f"all_angles_{self.droplet_geometry}.npy"
                ),
                np.array(angles),
            )
        print("List of contact angles by batch:", angles)
        return angles
