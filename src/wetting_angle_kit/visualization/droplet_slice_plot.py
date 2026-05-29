from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go


class DropletSlicePlotter:
    """Interactive Plotly slice visualization with toggleable layers."""

    def __init__(self, center: bool = True):
        """
        Parameters
        ----------
        center : bool, default True
            If True recentre z coordinates by subtracting mean wall z.
        """
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
        oxygen_position: np.ndarray,
        surface_data: list[np.ndarray],
        popt: Sequence[float],
        wall_coords: np.ndarray,
        alpha: float | None = None,
        y_com: float | None = None,
        pbc_y: float | None = None,
        show_water: bool = True,
        show_surface: bool = True,
        show_circle: bool = True,
        show_tangent: bool = True,
        show_wall: bool = True,
    ) -> Any:
        """Create interactive Plotly figure for a single frame slice.

        Parameters
        ----------
        oxygen_position : ndarray (N, 3)
            Oxygen atom coordinates.
        surface_data : list[array]
            List of surface contours for selected slice.
        popt : sequence
            Fitted circle parameters (x_center, z_center, radius, extra).
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
            x_center, z_center, radius, _ = popt
            z_center -= z_shift
        else:
            x_center, z_center, radius, _ = popt
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
                        line=dict(color=self.surface_color, width=3),
                        visible=True,
                        showlegend=True,
                    )
                )
        # --- Fitted circle ---
        if show_circle:
            theta = np.linspace(0, 2 * np.pi, 200)
            circle_x = x_center + radius * np.cos(theta)
            circle_z = z_center + radius * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_z,
                    mode="lines",
                    name="Fitted Circle",
                    line=dict(color=self.circle_color, width=3, dash="dash"),
                    visible=True,
                    showlegend=True,
                )
            )
        # --- Tangent + α arc ---
        if show_tangent and alpha is not None:
            z_line = min([np.min(surf[:, 1]) for surf in surface_data])
            delta_z = z_line - z_center
            discriminant = radius**2 - delta_z**2
            if discriminant > 0:
                x_contact = x_center + np.sqrt(discriminant)  # Right side
                z_contact = z_line
                m_tangent = -(x_contact - x_center) / (z_contact - z_center)
                # Tangent line
                z_top = z_center + radius * 1.1
                x_top = x_contact + (z_top - z_contact) / m_tangent
                x_line = np.linspace(x_contact, x_top, 100)
                z_line_tan = m_tangent * (x_line - x_contact) + z_contact
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=z_line_tan,
                        mode="lines",
                        name=f"{alpha:.1f}°",
                        line=dict(color=self.tangent_color, width=3),
                        visible=True,
                        showlegend=True,
                    )
                )
                # α arc (left side)
                alpha_rad = np.radians(alpha)
                arc_radius = radius * 0.25
                theta_arc = np.linspace(np.pi - alpha_rad, np.pi, 100)
                arc_x = x_contact + arc_radius * np.cos(theta_arc)
                arc_z = z_contact + arc_radius * np.sin(theta_arc)
                fig.add_trace(
                    go.Scatter(
                        x=arc_x,
                        y=arc_z,
                        mode="lines",
                        name=f"{alpha:.1f}° Arc",
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
                y=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                itemsizing="constant",
                font=dict(size=10),
            ),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        return fig
