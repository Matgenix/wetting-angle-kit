import numpy as np
import plotly.graph_objects as go

from wetting_angle_kit.analysis.slicing import SlicingFrameFitter
from wetting_angle_kit.io_utils import recenter_droplet_pbc
from wetting_angle_kit.parsers import (
    LammpsDumpParser,
    LammpsDumpWallParser,
    LammpsDumpWaterFinder,
)
from wetting_angle_kit.visualization.droplet_slice_plot import DropletSlicePlotter


class LammpsContactAngleAnimator:
    """Plotly slider animation of the median per-frame slice angle.

    This class is **LAMMPS-specific**: it instantiates
    :class:`~wetting_angle_kit.parsers.LammpsDumpParser`,
    :class:`~wetting_angle_kit.parsers.LammpsDumpWallParser` and
    :class:`~wetting_angle_kit.parsers.LammpsDumpWaterFinder` directly. A
    parser-agnostic version would have to dispatch all three from a
    factory; the rename makes the current coupling explicit rather than
    promising generality that the implementation does not deliver.
    """

    def __init__(
        self,
        filename: str,
        oxygen_type: int,
        hydrogen_type: int,
        liquid_particle_types: set,
        n_frames: int = 10,
        droplet_geometry: str = "cylinder_y",
        delta_cylinder: float | None = None,
        delta_gamma: float | None = None,
        max_dist: int = 100,
        precentered: bool = False,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to LAMMPS dump trajectory file.
        oxygen_type : int
            LAMMPS particle type ID for oxygen atoms.
        hydrogen_type : int
            LAMMPS particle type ID for hydrogen atoms.
        liquid_particle_types : set
            LAMMPS particle type IDs for all liquid atoms (used to mask wall parser).
        n_frames : int, default 10
            Number of frames to include in the animation.
        droplet_geometry : str, default "cylinder_y"
            Droplet geometry passed to SlicingFrameFitter.
        delta_cylinder : float, optional
            Step size along the slicing axis (Å); required for
            ``cylinder_x`` / ``cylinder_y`` modes, must be None for spherical.
        delta_gamma : float, optional
            Azimuthal step (degrees) for spherical droplet geometry;
            required for spherical, must be None for cylinder modes.
        max_dist : int, default 100
            Maximum radial distance for line sampling (Å).
        precentered : bool, default False
            Set True if the trajectory already recenters the droplet at
            every frame and atoms are not wrapped across periodic
            boundaries; the per-frame circular-mean recentering is then
            skipped. Setting this on a trajectory that does NOT satisfy the
            precondition will misplace the contact-angle overlay.
        """
        if droplet_geometry == "spherical":
            if delta_gamma is None:
                raise ValueError("delta_gamma must be provided for spherical analysis")
            if delta_cylinder is not None:
                raise ValueError(
                    "delta_cylinder must not be set for spherical analysis "
                    "(it is only valid for cylinder_x / cylinder_y)."
                )
        elif droplet_geometry in ("cylinder_x", "cylinder_y"):
            if delta_cylinder is None:
                raise ValueError(
                    f"delta_cylinder must be provided for {droplet_geometry}."
                )
            if delta_gamma is not None:
                raise ValueError(
                    f"delta_gamma must not be set for {droplet_geometry} "
                    "(it is only valid for spherical)."
                )
        self.filename = filename
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.liquid_particle_types = liquid_particle_types
        self.n_frames = n_frames
        self.droplet_geometry = droplet_geometry
        self.delta_cylinder = delta_cylinder
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.precentered = precentered

        # Initialize objects
        self.wat_find = LammpsDumpWaterFinder(
            self.filename,
            oxygen_type=self.oxygen_type,
            hydrogen_type=self.hydrogen_type,
        )
        self.oxygen_indices = self.wat_find.get_water_oxygen_ids(frame_index=0)
        self.coord_wall = LammpsDumpWallParser(
            self.filename, liquid_particle_types=list(self.liquid_particle_types)
        )
        self.wall_coords = self.coord_wall.parse(frame_index=0)
        self.parser = LammpsDumpParser(filepath=self.filename)
        self.plotter = DropletSlicePlotter(center=True)

    def generate_animation(
        self, output_filename: str = "ContactAngle_Median_PerFrame_Slider.html"
    ) -> None:
        """Build and write HTML with slider of median contact angles over frames.

        Parameters
        ----------
        output_filename : str, default "ContactAngle_Median_PerFrame_Slider.html"
            Output HTML file path.
        """
        fig = go.Figure()
        frames_list = []
        frame_labels = []
        median_angles = []
        for frame_idx in range(self.n_frames):
            oxygen_position = self.parser.parse(
                frame_index=frame_idx, indices=self.oxygen_indices
            )
            if self.precentered:
                liquid_geom_center = np.mean(oxygen_position, axis=0)
            else:
                box_size_xy = (
                    self.parser.box_size_x(frame_index=frame_idx),
                    self.parser.box_size_y(frame_index=frame_idx),
                )
                oxygen_position, liquid_geom_center = recenter_droplet_pbc(
                    oxygen_position, self.droplet_geometry, box_size=box_size_xy
                )
            processor = SlicingFrameFitter(
                liquid_coordinates=oxygen_position,
                liquid_geom_center=liquid_geom_center,
                droplet_geometry=self.droplet_geometry,
                delta_cylinder=self.delta_cylinder,
                delta_gamma=self.delta_gamma,
                max_dist=self.max_dist,
            )
            angles, surfaces, popt_arrays = processor.predict_contact_angle()
            if not angles:
                # No slice in this frame produced a usable contact angle
                # (e.g. fitting failed everywhere). Skip the frame rather
                # than letting the median lookup crash on an empty list.
                continue
            median_idx = np.argsort(angles)[len(angles) // 2]
            alpha = angles[median_idx]
            popt = popt_arrays[median_idx]
            surface = [surfaces[median_idx]]
            median_angles.append(alpha)
            fig_frame = self.plotter.plot_surface_points(
                oxygen_position=oxygen_position,
                surface_data=surface,
                popt=popt,
                wall_coords=self.wall_coords.copy(),
                y_com=np.mean(oxygen_position[:, 1]),
                pbc_y=None,
                alpha=alpha,
                show_water=True,
                show_surface=True,
                show_circle=True,
                show_tangent=True,
                show_wall=True,
            )
            frame = go.Frame(
                data=fig_frame.data,
                name=f"Frame {frame_idx}",
                layout=go.Layout(
                    title_text=(
                        f"Frame {frame_idx} | Median contact angle = {alpha:.2f}°"
                    )
                ),
            )
            frames_list.append(frame)
            frame_labels.append(f"Frame {frame_idx}")
        if not frames_list:
            raise RuntimeError(
                "No frame produced a usable contact angle; cannot build animation."
            )
        fig.frames = frames_list
        fig.add_traces(frames_list[0].data)
        fig.update_layout(
            title="Interactive Contact Angle Evolution (Median Slice per Frame)",
            width=800,
            height=600,
            margin=dict(l=80, r=200, t=80, b=100),
            xaxis_title="x (Å)",
            yaxis_title="z (Å)",
            template="simple_white",
            showlegend=True,
            legend=dict(
                x=1.05,
                y=0.95,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=11),
            ),
            xaxis=dict(
                mirror=True,
                showline=True,
                linecolor="black",
                ticks="outside",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
            ),
            yaxis=dict(
                mirror=True,
                showline=True,
                linecolor="black",
                ticks="outside",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            sliders=[
                {
                    "active": 0,
                    "pad": {"b": 60, "t": 40},
                    "x": 0.2,
                    "len": 0.6,
                    "y": -0.1,
                    "yanchor": "top",
                    "steps": [
                        {
                            "args": [
                                [f"Frame {k}"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f"{k}",
                            "method": "animate",
                        }
                        for k in range(len(frames_list))
                    ],
                }
            ],
        )
        fig.write_html(output_filename)
