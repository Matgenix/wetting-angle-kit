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


class ContactAngleAnimator:
    """Generate interactive Plotly slider animation of median slice angle per frame."""

    def __init__(
        self,
        filename: str,
        particle_type_wall: set,
        oxygen_type: int,
        hydrogen_type: int,
        liquid_particle_types: set,
        n_frames: int = 10,
        droplet_geometry: str = "cylinder_y",
        delta_cylinder: int = 5,
        max_dist: int = 100,
        width_cylinder: int = 21,
        precentered: bool = False,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to LAMMPS dump trajectory file.
        particle_type_wall : set
            LAMMPS particle type IDs for wall atoms.
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
        delta_cylinder : int, default 5
            Step size along the slicing axis (Å).
        max_dist : int, default 100
            Maximum radial distance for line sampling (Å).
        width_cylinder : int, default 21
            Box extent along the cylinder axis (Å).
        precentered : bool, default False
            Set True if the trajectory already recenters the droplet at
            every frame and atoms are not wrapped across periodic
            boundaries; the per-frame circular-mean recentering is then
            skipped. Setting this on a trajectory that does NOT satisfy the
            precondition will misplace the contact-angle overlay.
        """
        self.filename = filename
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.liquid_particle_types = liquid_particle_types
        self.n_frames = n_frames
        self.droplet_geometry = droplet_geometry
        self.delta_cylinder = delta_cylinder
        self.max_dist = max_dist
        self.width_cylinder = width_cylinder
        self.precentered = precentered

        # Initialize objects
        self.wat_find = LammpsDumpWaterFinder(
            self.filename,
            particle_type_wall=self.particle_type_wall,
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
                max_dist=self.max_dist,
                width_cylinder=self.width_cylinder,
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
