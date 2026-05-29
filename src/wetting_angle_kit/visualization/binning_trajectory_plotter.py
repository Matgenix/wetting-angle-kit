from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from wetting_angle_kit.analysis.binning.results import BinningResults
from wetting_angle_kit.visualization.base_trajectory_plotter import (
    BaseTrajectoryPlotter,
)
from wetting_angle_kit.visualization.stats import TrajectoryStats


class BinningTrajectoryPlotter(BaseTrajectoryPlotter):
    """Plot statistics derived from one or more :class:`BinningResults`."""

    @staticmethod
    def circular_segment_area(R: float, z_center: float, z_cut: float) -> float:
        """Area of the circular cap of radius ``R`` below height ``z_cut``."""
        h = (z_center + R) - z_cut
        if h <= 0:
            return 0.0
        if h >= 2 * R:
            return float(np.pi * R**2)
        if h <= R:
            return float(
                R**2 * np.arccos((R - h) / R) - (R - h) * np.sqrt(2 * R * h - h**2)
            )
        h_small = 2 * R - h
        return float(
            np.pi * R**2
            - (
                R**2 * np.arccos((R - h_small) / R)
                - (R - h_small) * np.sqrt(2 * R * h_small - h_small**2)
            )
        )

    def __init__(
        self,
        results: BinningResults | Iterable[BinningResults],
        labels: list[str] | None = None,
        time_steps: list[float] | None = None,
        time_unit: str = "ps",
    ) -> None:
        """
        Parameters
        ----------
        results : BinningResults or iterable of BinningResults
            One results container per trajectory.
        labels : list of str, optional
            Display labels (one per results container). Defaults to
            ``["trajectory_0", ...]``.
        time_steps : list of float, optional
            Per-trajectory time step applied to ``batch_index`` for the
            time axis of evolution plots. Defaults to ``1.0`` for each.
        time_unit : str, optional
            Time unit shown on x-axis labels.
        """
        if isinstance(results, BinningResults):
            results = [results]
        else:
            results = list(results)
        self.results = results
        self.labels = labels or [f"trajectory_{i}" for i in range(len(results))]
        self.time_steps = time_steps or [1.0] * len(results)
        self.time_unit = time_unit

    def _surface_areas(self, result: BinningResults) -> np.ndarray:
        """Per-batch circular-cap surface area from fitted (R_eq, zi_c, zi_0)."""
        return np.array(
            [
                self.circular_segment_area(
                    batch.fitted_params["R_eq"],
                    batch.fitted_params["zi_c"],
                    batch.fitted_params["zi_0"],
                )
                for batch in result.batches
            ]
        )

    def summary(self) -> list[TrajectoryStats]:
        stats: list[TrajectoryStats] = []
        for label, result in zip(self.labels, self.results, strict=False):
            surfaces = self._surface_areas(result)
            stats.append(
                TrajectoryStats(
                    method_name="Binning Analysis",
                    label=label,
                    mean_surface_area=float(np.mean(surfaces)),
                    mean_contact_angle=result.mean_angle,
                    std_contact_angle=result.std_angle,
                    n_samples=len(result),
                )
            )
        return stats

    def plot_angle_evolution(self, save_path: str | None = None) -> go.Figure:
        """Plot per-batch contact angle as a function of batch time.

        Parameters
        ----------
        save_path : str, optional
            If provided, write the figure as standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with one line per trajectory.
        """
        fig = go.Figure()
        for label, result, dt in zip(
            self.labels, self.results, self.time_steps, strict=False
        ):
            times = np.array([b.batch_index for b in result.batches]) * dt
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=result.angles_per_batch,
                    mode="lines+markers",
                    name=label,
                    line=dict(width=2),
                )
            )
        fig.update_layout(
            title="Contact angle evolution (per batch)",
            xaxis_title=f"Batch time ({self.time_unit})",
            yaxis_title="Contact angle (°)",
            template="plotly_white",
        )
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_density_contour(
        self,
        result_index: int = 0,
        batch_index: int = 0,
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot the density field of one batch with the fitted isoline.

        Parameters
        ----------
        result_index : int, default 0
            Index into the results list (selects which trajectory).
        batch_index : int, default 0
            Index of the batch within that trajectory.
        save_path : str, optional
            If provided, write the figure as standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            Filled contour of the density field plus dashed circle / wall
            isoline traces when available.
        """
        batch = self.results[result_index].batches[batch_index]
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=batch.xi_cc,
                y=batch.zi_cc,
                z=np.transpose(batch.rho_cc),
                colorscale="Jet",
                colorbar=dict(title="ρ"),
            )
        )
        if batch.circle_xi is not None and batch.circle_zi is not None:
            fig.add_trace(
                go.Scatter(
                    x=batch.circle_xi,
                    y=batch.circle_zi,
                    mode="lines",
                    name="Fitted droplet",
                    line=dict(color="black", dash="dash", width=2),
                )
            )
        if batch.wall_line_xi is not None and batch.wall_line_zi is not None:
            fig.add_trace(
                go.Scatter(
                    x=batch.wall_line_xi,
                    y=batch.wall_line_zi,
                    mode="lines",
                    name="Fitted wall",
                    line=dict(color="black", dash="dash", width=2),
                )
            )
        fig.update_layout(
            title=(
                f"Density field — {self.labels[result_index]} "
                f"(batch {batch.batch_index})"
            ),
            xaxis_title="ξ (Å)",
            yaxis_title="z (Å)",
            template="plotly_white",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        if save_path:
            fig.write_html(save_path)
        return fig
