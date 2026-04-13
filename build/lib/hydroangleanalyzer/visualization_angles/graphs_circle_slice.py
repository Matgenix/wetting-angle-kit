import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.ticker import AutoMinorLocator

from wetting_angle_kit.contact_angle_method.sliced_method import ContactAngleSliced
from wetting_angle_kit.parser import (
    DumpParser,
    DumpWallParser,
    DumpWaterMoleculeFinder,
)

plt.style.use("seaborn-v0_8-whitegrid")


class DropletSlicedPlotter:
    """Plot droplet slice: surface contours, fitted circle and tangent line.

    Parameters
    ----------
    center : bool, default True
        If True recentre z coordinates by subtracting mean wall z.
    show_wall : bool, default True
        Whether to draw wall particles.
    molecule_view : bool, default True
        If True draw fake hydrogens around each oxygen (schematic water view).
    """

    def __init__(
        self, center: bool = True, show_wall: bool = True, molecule_view: bool = True
    ):
        self.center = center
        self.show_wall = show_wall
        self.molecule_view = molecule_view

        # Colors
        self.oxygen_color = "#d62828"
        self.hydrogen_color = "white"
        self.surface_color = "black"
        self.circle_color = "#0A9396"
        self.wall_color = "black"
        self.tangent_color = "#bb3e03"

    def plot_surface_points(
        self,
        oxygen_position,
        surface_data,
        popt,
        wall_coords=None,
        output_filename=None,
        y_com=None,
        pbc_y=None,
        alpha=None,
    ):
        """Render slice figure and save to file.

        Parameters
        ----------
        oxygen_position : ndarray (N, 3)
            Cartesian coordinates of oxygen atoms for the frame.
        surface_data : list[array]
            List of arrays with surface line coordinates (x, z) for each slice.
        popt : sequence
            Fitted circle parameters (Xc, Zc, R, extra) for chosen slice.
        wall_coords : ndarray (M, 3)
            Wall particle coordinates.
        output_filename : str or Path
            Path to save the PNG figure.
        y_com : float, optional
            Y centre used to select atoms in a thin slice. If None computed.
        pbc_y : float, optional
            Box length in y for PBC wrapping; if provided shortest-distance used.
        alpha : float, optional
            Contact angle in degrees; if given draw tangent line and arc.

        Returns
        -------
        None
            Saves figure to ``output_filename`` and closes it.
        """

        if y_com is None:
            y_com = np.mean(oxygen_position[:, 1])

        # Select atoms near the Y center (±3 Å)
        if pbc_y is not None:
            dy = np.abs(oxygen_position[:, 1] - y_com)
            dy = np.minimum(dy, pbc_y - dy)
            mask = dy <= 3
        else:
            mask = np.abs(oxygen_position[:, 1] - y_com) <= 3
        oxygen_selected = oxygen_position[mask]

        # --- Subsample for clarity ---
        rng = np.random.default_rng(42)
        keep_fraction = 0.70
        sample_idx = rng.choice(
            len(oxygen_selected),
            size=int(len(oxygen_selected) * keep_fraction),
            replace=False,
        )
        oxygen_selected = oxygen_selected[sample_idx]

        # --- Limit wall region under droplet (±5 Å margin) ---
        x_min, x_max = (
            np.min(oxygen_selected[:, 0]) - 5,
            np.max(oxygen_selected[:, 0]) + 5,
        )

        # Only process wall_coords if needed
        if self.show_wall and wall_coords is not None:
            wall_mask = (wall_coords[:, 0] >= x_min) & (wall_coords[:, 0] <= x_max)
            wall_coords = wall_coords[wall_mask]

        # --- Optional recentring ---
        if self.center and wall_coords is not None:
            z_shift = np.mean(wall_coords[:, 2])
            oxygen_selected[:, 2] -= z_shift
            wall_coords[:, 2] -= z_shift
            surface_data = [
                np.column_stack([surf[:, 0], surf[:, 1] - z_shift])
                for surf in surface_data
            ]
            Xc, Zc, R, limit_med = popt
            Zc -= z_shift
        else:
            Xc, Zc, R, limit_med = popt

        # --- Plot setup ---
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)

        # --- Wall atoms ---
        if self.show_wall:
            ax.scatter(
                wall_coords[:, 0],
                wall_coords[:, 2],
                color=self.wall_color,
                s=3,
                alpha=0.7,
                zorder=0,
            )

        # --- Water representation ---
        if self.molecule_view:
            h_dist = 1.0
            for ox, _oy, oz in oxygen_selected:
                ax.scatter(
                    ox,
                    oz,
                    color=self.oxygen_color,
                    s=8,
                    alpha=0.9,
                    edgecolors="none",
                    linewidths=0.15,
                    zorder=1,
                )
                for _ in range(2):
                    angle = rng.uniform(0, 2 * np.pi)
                    dx, dz = h_dist * np.cos(angle), h_dist * np.sin(angle)
                    ax.scatter(
                        ox + dx,
                        oz + dz,
                        color=self.hydrogen_color,
                        s=4,
                        alpha=0.8,
                        edgecolors="black",
                        linewidths=0.15,
                        zorder=1,
                    )
        else:
            ax.scatter(
                oxygen_selected[:, 0],
                oxygen_selected[:, 2],
                color=self.oxygen_color,
                s=6,
                alpha=0.9,
                zorder=1,
            )

        # --- Surface line ---
        for surf in surface_data:
            X_data, Z_data = surf[:, 0], surf[:, 1]
            if not np.allclose([X_data[0], Z_data[0]], [X_data[-1], Z_data[-1]]):
                X_data = np.append(X_data, X_data[0])
                Z_data = np.append(Z_data, Z_data[0])
            ax.plot(X_data, Z_data, color=self.surface_color, lw=1.5, zorder=3)

        # --- Fitted circle ---
        circle = plt.Circle(
            (Xc, Zc), R, color=self.circle_color, fill=False, ls="--", lw=2.5, zorder=4
        )
        ax.add_artist(circle)
        # --- Tangent line (based on circle–surface intersection) ---
        if alpha is not None:
            alpha_rad = np.radians(alpha)

            # --- Determine the contact point from the surface bottom ---
            z_line = min([np.min(surf[:, 1]) for surf in surface_data])
            Xc, Zc, R, _ = popt

            delta_z = z_line - Zc
            discriminant = R**2 - delta_z**2
            if discriminant <= 0:
                return

            dx = np.sqrt(discriminant)

            # Choose correct side (right if α > 90°, left if α < 90°)
            if alpha > 90:
                x_contact = Xc + dx
            else:
                x_contact = Xc - dx
            z_contact = z_line

            # --- Tangent slope at the intersection point ---
            m_tangent = -(x_contact - Xc) / (z_contact - Zc)

            # --- Extend tangent line upwards to top of circle ---
            z_top = Zc + R * 1.1  # extend slightly above for visibility
            if abs(m_tangent) > 1e-6:
                x_top = x_contact + (z_top - z_contact) / m_tangent
            else:
                x_top = x_contact
            x_line = np.linspace(x_contact, x_top, 100)
            z_line = m_tangent * (x_line - x_contact) + z_contact

            # Draw tangent line
            ax.plot(
                x_line,
                z_line,
                color=self.tangent_color,
                lw=2.0,
                ls="-",
                label=f"Tangent (α={alpha:.1f}°)",
                zorder=5,
            )

            # --- Parameters from the circle fit ---
            Xc, Zc, R = popt[:3]

            # --- Determine intersection (right side only) ---
            z_line = min([np.min(surf[:, 1]) for surf in surface_data])
            delta_z = z_line - Zc
            discriminant = R**2 - delta_z**2
            if discriminant <= 0:
                return

            x_contact = Xc + np.sqrt(discriminant)  # right-side intersection
            z_contact = z_line

            # --- Tangent slope at contact point ---
            m_tangent = -(x_contact - Xc) / (z_contact - Zc)

            # --- Draw tangent line up to top of circle ---
            z_top = Zc + R * 1.1
            x_top = x_contact + (z_top - z_contact) / m_tangent
            x_line = np.linspace(x_contact, x_top, 100)
            z_line_tan = m_tangent * (x_line - x_contact) + z_contact

            ax.plot(
                x_line,
                z_line_tan,
                color="tab:orange",
                lw=2.0,
                label=f"Tangent (α={alpha:.1f}°)",
                zorder=5,
            )

            # --- Draw arc centered at contact point (right side) ---
            alpha_rad = np.radians(alpha)
            arc_radius = R * 0.25
            theta = np.linspace(
                np.pi - alpha_rad, np.pi, 100
            )  # from horizontal (0) to tangent (α)
            arc_x = x_contact + arc_radius * np.cos(theta)
            arc_z = z_contact + arc_radius * np.sin(theta)
            ax.plot(arc_x, arc_z, color="gray", lw=1.5, zorder=6)

            # --- Label α value near the middle of the arc ---
            mid_theta = alpha_rad / 2
            text_x = x_contact + 1.2 * arc_radius * np.cos(mid_theta)
            text_z = z_contact + 1.2 * arc_radius * np.sin(mid_theta)
            ax.text(
                text_x,
                text_z,
                f"{alpha:.1f}°",
                fontsize=9,
                color="black",
                ha="center",
                va="center",
                zorder=7,
            )

        # --- Axes ---
        ax.set_xlabel("x (Å)", fontsize=9)
        ax.set_ylabel("z (Å)", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xlim(x_min - 5, x_max + 5)

        # --- Legend ---
        ax.legend(
            handles=[
                plt.Line2D(
                    [], [], color=self.surface_color, lw=1.5, label="Surface contour"
                ),
                plt.Line2D(
                    [],
                    [],
                    color=self.circle_color,
                    ls="--",
                    lw=1.5,
                    label="Fitted circle",
                ),
                plt.Line2D(
                    [], [], color=self.tangent_color, lw=1.5, label="Tangent line"
                ),
            ],
            loc="upper left",
            frameon=False,
            fontsize=7,
        )

        plt.tight_layout(pad=0.1)
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()


class DropletSlicedPlotterPlotly:
    """Interactive Plotly slice visualization with toggleable layers."""

    def __init__(self, center: bool = True):
        self.center = center
        # Colors
        self.oxygen_color = "#d62828"
        self.hydrogen_color = "#FFFFFF"
        self.surface_color = "#000000"
        self.circle_color = "#0A9396"
        self.wall_color = "#000000"
        self.tangent_color = "#bb3e03"

    def plot_surface_points(
        self,
        oxygen_position,
        surface_data,
        popt,
        wall_coords,
        alpha=None,
        y_com=None,
        pbc_y=None,
        show_water=True,
        show_surface=True,
        show_circle=True,
        show_tangent=True,
        show_wall=True,
    ):
        """Create interactive Plotly figure for a single frame slice.

        Parameters
        ----------
        oxygen_position : ndarray (N, 3)
            Oxygen atom coordinates.
        surface_data : list[array]
            List of surface contours for selected slice.
        popt : sequence
            Fitted circle parameters (Xc, Zc, R, extra).
        wall_coords : ndarray (M, 3)
            Wall particle coordinates.
        alpha : float, optional
            Contact angle for tangent construction.
        y_com : float, optional
            Mean y used for slicing; computed if None.
        pbc_y : float, optional
            Y box length for periodic slicing.
        show_water, show_surface, show_circle, show_tangent, show_wall : bool
            Layer visibility toggles.

        Returns
        -------
        plotly.graph_objects.Figure
            Configured figure object (not saved).
        """
        if y_com is None:
            y_com = np.mean(oxygen_position[:, 1])
        # Select slice in y-direction
        if pbc_y is not None:
            dy = np.abs(oxygen_position[:, 1] - y_com)
            dy = np.minimum(dy, pbc_y - dy)
            mask = dy <= 3
        else:
            mask = np.abs(oxygen_position[:, 1] - y_com) <= 3
        oxygen_selected = oxygen_position[mask]
        # Recenter if needed
        if self.center:
            z_shift = np.mean(wall_coords[:, 2])
            oxygen_selected[:, 2] -= z_shift
            wall_coords[:, 2] -= z_shift
            surface_data = [
                np.column_stack([surf[:, 0], surf[:, 1] - z_shift])
                for surf in surface_data
            ]
            Xc, Zc, R, _ = popt
            Zc -= z_shift
        else:
            Xc, Zc, R, _ = popt
        fig = go.Figure()
        # --- Wall ---
        if show_wall:
            fig.add_trace(
                go.Scatter(
                    x=wall_coords[:, 0],
                    y=wall_coords[:, 2],
                    mode="markers",
                    name="Wall",
                    marker=dict(color=self.wall_color, size=3),
                    opacity=0.7,
                    visible=True,
                    showlegend=True,
                )
            )
        # --- Water molecules ---
        if show_water:
            fig.add_trace(
                go.Scatter(
                    x=oxygen_selected[:, 0],
                    y=oxygen_selected[:, 2],
                    mode="markers",
                    name="Water",
                    marker=dict(color=self.oxygen_color, size=5),
                    opacity=0.8,
                    visible=True,
                    showlegend=True,
                )
            )
        # --- Surface contour ---
        if show_surface:
            for surf in surface_data:
                # Append the first point to the end to close the contour
                closed_surf = np.vstack([surf, surf[0]])
                fig.add_trace(
                    go.Scatter(
                        x=closed_surf[:, 0],
                        y=closed_surf[:, 1],
                        mode="lines",
                        name="Surface contour",
                        line=dict(color=self.surface_color, width=3),  # Thicker line
                        visible=True,
                        showlegend=True,
                    )
                )
        # --- Fitted circle ---
        if show_circle:
            theta = np.linspace(0, 2 * np.pi, 200)
            circle_x = Xc + R * np.cos(theta)
            circle_z = Zc + R * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_z,
                    mode="lines",
                    name="Fitted Circle",
                    line=dict(
                        color=self.circle_color, width=3, dash="dash"
                    ),  # Thicker line
                    visible=True,
                    showlegend=True,
                )
            )
        # --- Tangent + α arc ---
        if show_tangent and alpha is not None:
            z_line = min([np.min(surf[:, 1]) for surf in surface_data])
            delta_z = z_line - Zc
            discriminant = R**2 - delta_z**2
            if discriminant > 0:
                x_contact = Xc + np.sqrt(discriminant)  # Right side
                z_contact = z_line
                m_tangent = -(x_contact - Xc) / (z_contact - Zc)
                # Tangent line
                z_top = Zc + R * 1.1
                x_top = x_contact + (z_top - z_contact) / m_tangent
                x_line = np.linspace(x_contact, x_top, 100)
                z_line_tan = m_tangent * (x_line - x_contact) + z_contact
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=z_line_tan,
                        mode="lines",
                        name=f"{alpha:.1f}°",  # Only show angle value
                        line=dict(color=self.tangent_color, width=3),  # Thicker line
                        visible=True,
                        showlegend=True,
                    )
                )
                # α arc (left side)
                alpha_rad = np.radians(alpha)
                arc_radius = R * 0.25
                theta_arc = np.linspace(np.pi - alpha_rad, np.pi, 100)
                arc_x = x_contact + arc_radius * np.cos(theta_arc)
                arc_z = z_contact + arc_radius * np.sin(theta_arc)
                fig.add_trace(
                    go.Scatter(
                        x=arc_x,
                        y=arc_z,
                        mode="lines",
                        name=f"{alpha:.1f}° Arc",  # Only show angle value
                        line=dict(color="gray", width=2),
                        visible=True,
                        showlegend=False,
                    )
                )

                # Label α near mid-arc
                mid_theta = alpha_rad / 2
                text_x = x_contact + 1.2 * arc_radius * np.cos(mid_theta)
                text_z = z_contact + 1.2 * arc_radius * np.sin(mid_theta)
                fig.add_annotation(
                    x=text_x,
                    y=text_z,
                    text=f"{alpha:.1f}°",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                )
        # --- Layout ---
        fig.update_layout(
            width=600,
            height=450,
            xaxis_title="x (Å)",
            yaxis_title="z (Å)",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1,  # Position legend outside the plot
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                itemsizing="constant",  # Ensures checkboxes are clearly visible
                font=dict(size=10),
            ),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        return fig


class ContactAngleAnimator:
    """Generate interactive Plotly slider animation of median slice angle per frame."""

    def __init__(
        self,
        filename: str,
        particle_type_wall: set,
        oxygen_type: int,
        hydrogen_type: int,
        particule_liquid_type: set,
        n_frames: int = 10,
        droplet_geometry: str = "cylinder_y",
        delta_cylinder: int = 5,
        max_dist: int = 100,
        width_cylinder: int = 21,
    ):
        self.filename = filename
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.particule_liquid_type = particule_liquid_type
        self.n_frames = n_frames
        self.droplet_geometry = droplet_geometry
        self.delta_cylinder = delta_cylinder
        self.max_dist = max_dist
        self.width_cylinder = width_cylinder

        # Initialize objects
        self.wat_find = DumpWaterMoleculeFinder(
            self.filename,
            particle_type_wall=self.particle_type_wall,
            oxygen_type=self.oxygen_type,
            hydrogen_type=self.hydrogen_type,
        )
        self.oxygen_indices = self.wat_find.get_water_oxygen_ids(frame_index=0)
        self.coord_wall = DumpWallParser(
            self.filename, particule_liquid_type=self.particule_liquid_type
        )
        self.wall_coords = self.coord_wall.parse(frame_index=1)
        self.parser = DumpParser(filepath=self.filename)
        self.plotter = DropletSlicedPlotterPlotly(center=True)

    def generate_animation(
        self, output_filename: str = "ContactAngle_Median_PerFrame_Slider.html"
    ):
        """Build and write HTML with slider of median contact angles over frames.
        Parameters
        ----------
        output_filename : str, default "ContactAngle_Median_PerFrame_Slider.html"
            Output HTML file path.
        Returns
        -------
        None
            Writes HTML file and prints path.
        """
        fig = go.Figure()
        frames_list = []
        frame_labels = []
        median_angles = []
        for frame_idx in range(self.n_frames):
            oxygen_position = self.parser.parse(
                frame_index=frame_idx, indices=self.oxygen_indices
            )
            processor = ContactAngleSliced(
                liquid_coordinates=oxygen_position,
                liquid_geom_center=np.mean(oxygen_position, axis=0),
                droplet_geometry=self.droplet_geometry,
                delta_cylinder=self.delta_cylinder,
                max_dist=self.max_dist,
                width_cylinder=self.width_cylinder,
            )
            list_alfas, array_surfaces, array_popt = processor.predict_contact_angle()
            median_idx = np.argsort(list_alfas)[len(list_alfas) // 2]
            alpha = list_alfas[median_idx]
            popt = array_popt[median_idx]
            surface = np.array([array_surfaces[median_idx]])
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
                        f"Frame {frame_idx} | Median contact angle = "
                        f"{alpha:.2f}\u00b0"
                    )
                ),
            )
            frames_list.append(frame)
            frame_labels.append(f"Frame {frame_idx}")
        fig.frames = frames_list
        fig.add_traces(frames_list[0].data)
        fig.update_layout(
            title=("Interactive Contact Angle Evolution (Median Slice per " "Frame)"),
            width=800,
            height=600,
            margin=dict(l=80, r=200, t=80, b=100),
            xaxis_title="x (\u03c3)",
            yaxis_title="z (\u03c3)",
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
        print(f"Interactive HTML saved: {output_filename}")


# Example usage
# if __name__ == "__main__":
#    animator = ContactAngleAnimator(
#        filename="../wetting_angle_kit/tests/trajectories/"
#        "traj_10_3_330w_nve_4k_reajust.lammpstrj",
#        particle_type_wall={3},
#        oxygen_type=1,
#        hydrogen_type=2,
#        particule_liquid_type={2, 1},
#        n_frames=10,
#    )
#    animator.generate_animation()
