from collections.abc import Iterable

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from wetting_angle_kit.analysis.slicing.results import SlicingResults
from wetting_angle_kit.visualization.base_trajectory_plotter import (
    BaseTrajectoryPlotter,
)
from wetting_angle_kit.visualization.stats import TrajectoryStats


def _shoelace_area(points: np.ndarray) -> float:
    """Polygon area via the shoelace formula."""
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Return a CSS ``rgba(...)`` string from a ``#rrggbb`` hex color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


class SlicingTrajectoryPlotter(BaseTrajectoryPlotter):
    """Plot statistics derived from one or more :class:`SlicingResults`."""

    def __init__(
        self,
        results: SlicingResults | Iterable[SlicingResults],
        labels: list[str] | None = None,
        time_steps: list[float] | None = None,
        time_unit: str = "ps",
    ) -> None:
        """
        Parameters
        ----------
        results : SlicingResults or iterable of SlicingResults
            One results container per trajectory.
        labels : list of str, optional
            Display labels (one per results container). Defaults to
            ``["trajectory_0", ...]``.
        time_steps : list of float, optional
            Per-trajectory time step applied to ``frames`` for the time
            axis of evolution plots. Defaults to ``1.0`` for each.
        time_unit : str, optional
            Time unit shown on x-axis labels.
        """
        if isinstance(results, SlicingResults):
            results = [results]
        else:
            results = list(results)
        self.results = results
        self.labels = labels or [f"trajectory_{i}" for i in range(len(results))]
        self.time_steps = time_steps or [1.0] * len(results)
        self.time_unit = time_unit

    def _mean_surface_areas(self, result: SlicingResults) -> np.ndarray:
        """Per-frame mean polygon area (shoelace over the frame's slices)."""
        return np.array(
            [
                float(np.mean([_shoelace_area(s) for s in frame_surfaces]))
                for frame_surfaces in result.surfaces
            ]
        )

    def summary(self) -> list[TrajectoryStats]:
        stats: list[TrajectoryStats] = []
        for label, result in zip(self.labels, self.results, strict=False):
            surfaces = self._mean_surface_areas(result)
            stats.append(
                TrajectoryStats(
                    method_name="Slicing Analysis",
                    label=label,
                    mean_surface_area=float(np.mean(surfaces)),
                    mean_contact_angle=result.mean_angle,
                    std_contact_angle=result.std_angle,
                    n_samples=len(result),
                )
            )
        return stats

    def plot_angle_evolution(
        self,
        stat: str = "median",
        per_frame_std: bool = True,
        running_mean: bool = True,
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot per-frame contact angle as a function of time.

        Parameters
        ----------
        stat : str, default "median"
            Per-frame aggregation across slices; one of ``"median"`` or ``"mean"``.
        per_frame_std : bool, default True
            If True, draw a transparent ±σ band around the per-frame curve
            using the inter-slice spread within each frame — shows how noisy
            the contact angle estimate is at each instant.
        running_mean : bool, default True
            If True, overlay the cumulative running mean of the per-frame
            central tendency as a dashed line, plus a transparent ±σ band
            of that cumulative series — shows how the time-averaged contact
            angle converges as more frames are accumulated.
        save_path : str, optional
            If provided, write the figure as standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with one per-frame line per trajectory, optionally with
            an inter-slice ±σ band and/or a running mean line with its
            cumulative ±σ band.
        """
        if stat not in ("median", "mean"):
            raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
        agg = np.median if stat == "median" else np.mean
        palette = pc.qualitative.Plotly
        band_traces: list[go.Scatter] = []
        line_traces: list[go.Scatter] = []
        for idx, (label, result, dt) in enumerate(
            zip(self.labels, self.results, self.time_steps, strict=False)
        ):
            color = palette[idx % len(palette)]
            band_color = _hex_to_rgba(color, 0.2)
            times = np.array(result.frames) * dt
            per_frame = np.array([float(agg(a)) for a in result.angles])
            per_frame_group = label
            running_group = f"{label} running mean"
            line_traces.append(
                go.Scatter(
                    x=times,
                    y=per_frame,
                    mode="lines",
                    name=label,
                    line=dict(width=2, color=color),
                    legendgroup=per_frame_group,
                )
            )
            if per_frame_std:
                std = np.array([float(np.std(a)) for a in result.angles])
                band_traces.append(
                    go.Scatter(
                        x=np.concatenate([times, times[::-1]]),
                        y=np.concatenate([per_frame + std, (per_frame - std)[::-1]]),
                        fill="toself",
                        fillcolor=band_color,
                        line=dict(width=0),
                        name=f"{label} ±σ",
                        legendgroup=per_frame_group,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            if running_mean:
                counts = np.arange(1, len(per_frame) + 1)
                cum_mean = np.cumsum(per_frame) / counts
                sq_mean = np.cumsum(per_frame**2) / counts
                cum_std = np.sqrt(np.maximum(sq_mean - cum_mean**2, 0.0))
                band_traces.append(
                    go.Scatter(
                        x=np.concatenate([times, times[::-1]]),
                        y=np.concatenate(
                            [cum_mean + cum_std, (cum_mean - cum_std)[::-1]]
                        ),
                        fill="toself",
                        fillcolor=band_color,
                        line=dict(width=0),
                        name=f"{label} running ±σ",
                        legendgroup=running_group,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                line_traces.append(
                    go.Scatter(
                        x=times,
                        y=cum_mean,
                        mode="lines",
                        name=running_group,
                        line=dict(width=2, color=color, dash="dash"),
                        legendgroup=running_group,
                    )
                )
        fig = go.Figure()
        for trace in band_traces:
            fig.add_trace(trace)
        for trace in line_traces:
            fig.add_trace(trace)
        fig.update_layout(
            title=f"Contact angle evolution ({stat})",
            xaxis_title=f"Time ({self.time_unit})",
            yaxis_title="Contact angle (°)",
            template="plotly_white",
        )
        if save_path:
            fig.write_html(save_path)
        return fig
