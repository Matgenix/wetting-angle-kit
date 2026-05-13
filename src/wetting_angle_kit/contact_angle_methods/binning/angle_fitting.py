"""Binning-method contact-angle analyzer.

Algorithm
---------

The trajectory is aggregated into a 2D density field ``rho(xi, zi)`` on a
regular bin grid, where ``xi`` is the in-plane radial coordinate produced
by :func:`wetting_angle_kit.parsers.base.project_to_profile` and
``zi`` is the lab-frame vertical coordinate. The histogram uses
:func:`numpy.histogram2d` (left-edge inclusive, right-edge exclusive,
last bin closed on both ends).

Per-bin volume elements:

* ``cylinder_x`` / ``cylinder_y``: ``dV = 2 * width_cylinder * dxi * dzi``.
  The factor of 2 accounts for folding the symmetric distribution into
  positive ``xi`` via ``|x_centered|``.
* ``spherical``: ``dV = 2 * pi * xi_cc * dxi * dzi`` — the annular shell
  volume of cylindrical coordinates.

A :class:`HyperbolicTangentModel` is then fitted to the time-averaged
density field and the implied contact angle is derived from the fitted
sphere radius, center, and wall position. Lengths are in Å, densities in
particles · Å⁻³, and the final contact angle is returned in degrees.
"""

import logging
import os
import warnings
from typing import Any

import matplotlib
import numpy as np

logger = logging.getLogger(__name__)

# Force a non-interactive backend before pyplot is imported so figure
# generation works in headless environments (CI, OVITO subprocesses).
# Only switch if no backend is already attached to an open figure.
if matplotlib.get_backend().lower() != "agg":
    try:
        matplotlib.use("Agg", force=False)
    except (ImportError, ValueError):
        # Backend is already locked in by another import; keep it.
        pass

import matplotlib.pyplot as plt  # noqa: E402

from wetting_angle_kit.contact_angle_methods.binning.surface_definition import (  # noqa: E402
    HyperbolicTangentModel,
)
from wetting_angle_kit.io_utils import validate_droplet_geometry  # noqa: E402


class ContactAngleBinning:
    """Binning-based contact angle estimator using density field fitting.

    Frames aggregated in spatial bins form a time-averaged density field.
    A hyperbolic tangent interface model is fitted and the implied contact
    angle is computed from fitted geometric parameters.
    """

    def __init__(
        self,
        parser: Any,
        atom_indices: Any,
        droplet_geometry: str = "spherical",
        width_cylinder: float | None = None,
        binning_params: dict[str, Any] | None = None,
        output_dir: str = "output_analysis/",
        plot_graphs: bool = True,
    ) -> None:
        validate_droplet_geometry(droplet_geometry)
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
                            parser.box_size_y(frame_index=0),
                            parser.box_size_x(frame_index=0),
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
            warnings.warn(
                "binning_params was not supplied; using a heuristic default "
                f"(xi_0=0, xi_f={max_dist}, zi_0=0, zi_f={max_dist}, "
                "50x50 bins) derived from one third of the largest in-plane "
                "box dimension. For accurate density fields, supply "
                "system-specific binning_params matching your droplet size "
                "and per-frame sampling.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.binning_params = binning_params
        self._initialize_grid()
        if self.width_cylinder is None:
            if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
                if self.droplet_geometry == "cylinder_x":
                    self.width_cylinder = self.parser.box_size_x(frame_index=0)
                elif self.droplet_geometry == "cylinder_y":
                    self.width_cylinder = self.parser.box_size_y(frame_index=0)
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_grid(self) -> None:
        """Initialize bin edges, centers and cell sizes from parameters."""
        self.xi = np.linspace(
            self.binning_params["xi_0"],
            self.binning_params["xi_f"],
            int(self.binning_params["nbins_xi"]),
        )
        self.zi = np.linspace(
            self.binning_params["zi_0"],
            self.binning_params["zi_f"],
            int(self.binning_params["nbins_zi"]),
        )
        self.dxi = self.xi[1] - self.xi[0]
        self.dzi = self.zi[1] - self.zi[0]
        self.xi_cc = 0.5 * (self.xi[1:] + self.xi[:-1])
        self.zi_cc = 0.5 * (self.zi[1:] + self.zi[:-1])

    def binning(
        self, xi_par: np.ndarray, zi_par: np.ndarray, len_frames: int
    ) -> np.ndarray:
        """Return 2D density field by binning particle coordinates.

        Uses :func:`numpy.histogram2d`, which is vectorized (O(N) in the
        particle count) and correctly handles particles on bin edges
        (inclusive on the left/lower edge, inclusive on the right/upper
        edge of the last bin only). This makes the legacy ``+0.01`` shift
        on the radial coordinate unnecessary.

        Parameters
        ----------
        xi_par : ndarray
            Radial/in-plane coordinate values for particles over frames.
        zi_par : ndarray
            Vertical coordinate values for particles over frames.
        len_frames : int
            Number of frames aggregated.

        Returns
        -------
        ndarray, shape (nbins_xi-1, nbins_zi-1)
            Averaged density field on cell centers.
        """
        counts, _, _ = np.histogram2d(
            xi_par,
            zi_par,
            bins=(self.xi, self.zi),  # type: ignore[arg-type]
        )  # shape (nbins_xi-1, nbins_zi-1)
        if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
            assert self.width_cylinder is not None
            dV = 2.0 * self.width_cylinder * self.dxi * self.dzi
            rho_cc = counts / dV
        else:  # spherical — annular shell volume depends on radius
            dV_per_row = 2.0 * np.pi * self.xi_cc * self.dxi * self.dzi
            rho_cc = counts / dV_per_row[:, np.newaxis]
        if len_frames > 0:
            rho_cc /= len_frames
        return rho_cc

    def plot_density_with_isoline(
        self,
        xi_cc: np.ndarray,
        zi_cc: np.ndarray,
        rho_cc: np.ndarray,
        circle_xi: np.ndarray,
        circle_zi: np.ndarray,
        wall_line_xi: np.ndarray,
        wall_line_zi: np.ndarray,
        batch_index: int | None = None,
        clevels: int = 20,
        scale: float = 0.75,
        close: bool = True,
    ) -> None:
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
        n_particles: float,
        param_strings: list[str],
        theta: float,
        xi_cc: np.ndarray,
        zi_cc: np.ndarray,
        rho_cc: np.ndarray,
        batch_index: int | None = None,
    ) -> None:
        """Write fitted parameters and density field CSV for a batch.

        Parameters
        ----------
        n_particles : float
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
            f.write(f"reduced_particles_number:{n_particles}\n")
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
        csv_data = np.c_[msh_xi_cc, msh_zi_cc, msh_rho_cc]
        np.savetxt(
            os.path.join(self.output_dir, f"rho_field{batch_str}.csv"),
            csv_data,
            delimiter=",",
            header=(f"x_{len(xi_cc)},y_{len(zi_cc)}," f"rho_{len(xi_cc) * len(zi_cc)}"),
        )

    def process_batch(
        self,
        frame_list: list[int],
        model: Any | None = None,
        batch_index: int | None = None,
    ) -> tuple[float, Any]:
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
        n_particles = len(xi_par) / max(len_frames, 1)
        batch_label = f" {batch_index}" if batch_index is not None else ""
        logger.info(
            f"Number of fluid particles in batch{batch_label}: {n_particles:.2f}"
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
        logger.info(
            f"Fitted parameters for batch{batch_label}:\n{''.join(param_strings)}"
        )
        contact_angle = model.compute_contact_angle()
        logger.info(f"Contact angle for batch{batch_label}: {contact_angle}")
        if self.plot_graphs:
            try:
                (
                    circle_xi,
                    circle_zi,
                    wall_line_xi,
                    wall_line_zi,
                ) = model.compute_isoline()
            except ValueError as exc:
                warnings.warn(
                    f"Skipping isoline plot for batch {batch_index}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
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
            n_particles,
            param_strings,
            contact_angle,
            self.xi_cc,
            self.zi_cc,
            rho_cc,
            batch_index,
        )
        return contact_angle, model

    def process_all_batches(
        self, batch_size: int = 100, save_angles: bool = True
    ) -> list[float]:
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
        frames_tot = self.parser.frame_count()
        logger.info(f"Total frames: {frames_tot}")
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
        logger.info(f"List of contact angles by batch: {angles}")
        return angles
