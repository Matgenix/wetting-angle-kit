"""Trajectory-level contact-angle evolution plot.

Mirrors the visual conventions of the legacy
``SlicingTrajectoryPlotter.plot_angle_evolution`` — a per-batch
contact-angle line with an optional inter-batch ``±σ`` band and a
cumulative running mean overlay — but consumes the new
:class:`TrajectoryResults` / :class:`CoupledFit2DResults` /
:class:`CoupledFit3DResults` shapes.

The plotter implements :class:`BaseTrajectoryPlotter`, so callers can
also fetch a :class:`TrajectoryStats` summary alongside the figure.
"""

from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit3DBatchResult,
    SlicingBatchResult,
    WholeBatchResult,
)
from wetting_angle_kit.visualization.base_trajectory_plotter import (
    BaseTrajectoryPlotter,
)
from wetting_angle_kit.visualization.stats import TrajectoryStats


def _shoelace_area(points: np.ndarray) -> float:
    """Polygon area via the shoelace formula."""
    if points.size == 0:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _circular_segment_area(R: float, z_center: float, z_cut: float) -> float:
    """Area of the circular segment of radius ``R`` above ``z_cut``."""
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


def _batch_surface_area(batch: Any) -> float:
    """Per-batch surface-area dispatch over the four result types."""
    if isinstance(batch, SlicingBatchResult):
        if not batch.slice_surfaces:
            return 0.0
        return float(np.mean([_shoelace_area(s) for s in batch.slice_surfaces]))
    if isinstance(batch, WholeBatchResult):
        popt = np.asarray(batch.popt)
        if popt.size == 5:  # spherical: [xc, yc, zc, R, z_wall]
            zc, R = float(popt[2]), float(popt[3])
        elif popt.size == 4:  # cylinder: [xc, zc, R, z_wall]
            zc, R = float(popt[1]), float(popt[2])
        else:
            return float("nan")
        return _circular_segment_area(R, zc, float(batch.z_wall))
    if isinstance(batch, (CoupledFit2DBatchResult, CoupledFit3DBatchResult)):
        params = batch.model_params
        return _circular_segment_area(
            float(params["R_eq"]),
            float(params["zi_c"]),
            float(params["zi_0"]),
        )
    return float("nan")


def _batch_central_angle(batch: Any, stat: Literal["mean", "median"]) -> float:
    """Per-batch central tendency.

    For slicing batches with per-slice angles, ``stat`` selects mean or
    median over the slices. For all other result shapes the only
    available scalar is :attr:`BatchResult.angle`, which is returned
    directly.
    """
    if isinstance(batch, SlicingBatchResult) and batch.per_slice_angles.size:
        if stat == "median":
            return float(np.median(batch.per_slice_angles))
        return float(np.mean(batch.per_slice_angles))
    return float(batch.angle)


class AngleEvolutionPlotter(BaseTrajectoryPlotter):
    """Plot per-batch contact-angle evolution across a trajectory.

    Parameters
    ----------
    results
        A ``*Results`` object exposing ``.batches`` with ``.angle`` and
        ``.frames`` (i.e. :class:`TrajectoryResults`,
        :class:`CoupledFit2DResults`, or
        :class:`CoupledFit3DResults`). For per-frame analyses use
        :class:`TemporalAggregator` with ``batch_size=1``; for pooled
        batches the evolution is shown per batch with each x-value at
        the pooled-frames midpoint.
    label : str, default ``"trajectory"``
        Display label used in legend entries and in
        :class:`TrajectoryStats`.
    timestep : float, default 1.0
        Time between two consecutive frames *in the trajectory file*
        (dump interval × MD integration timestep, not the integration
        timestep itself). Applied to ``frames`` to produce the x-axis.
    time_unit : str, default ``"ps"``
        Unit shown on the x-axis label.
    stat : {"median", "mean"}, default ``"median"``
        Per-batch central-tendency aggregation across slices for
        slicing results. Ignored for other result shapes (where only a
        single :attr:`BatchResult.angle` is defined).
    method_name : str, optional
        Free-form method tag used in :class:`TrajectoryStats`.
        Defaults to the underlying analyzer's class name when
        inferable.
    """

    def __init__(
        self,
        results: Any,
        *,
        label: str = "trajectory",
        timestep: float = 1.0,
        time_unit: str = "ps",
        stat: Literal["median", "mean"] = "median",
        method_name: str | None = None,
    ) -> None:
        if stat not in ("median", "mean"):
            raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
        self.results = results
        self.label = label
        self.timestep = timestep
        self.time_unit = time_unit
        self.stat = stat
        if method_name is None:
            method_name = type(results).__name__.replace("Results", "")
            if not method_name:
                method_name = "Analysis"
        self.method_name = method_name

    # ------------------------------------------------------------------
    # BaseTrajectoryPlotter interface.
    # ------------------------------------------------------------------

    def summary(self) -> list[TrajectoryStats]:
        batches = list(getattr(self.results, "batches", []))
        if not batches:
            return [
                TrajectoryStats(
                    method_name=self.method_name,
                    label=self.label,
                    mean_surface_area=float("nan"),
                    mean_contact_angle=float("nan"),
                    std_contact_angle=float("nan"),
                    n_samples=0,
                )
            ]
        areas = np.array([_batch_surface_area(b) for b in batches])
        return [
            TrajectoryStats(
                method_name=self.method_name,
                label=self.label,
                mean_surface_area=float(np.nanmean(areas)),
                mean_contact_angle=float(self.results.mean_angle),
                std_contact_angle=float(self.results.std_angle),
                n_samples=len(batches),
            )
        ]

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        per_frame_std: bool = True,
        running_mean: bool = True,
        title: str | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        """Build the angle-evolution figure.

        Parameters
        ----------
        per_frame_std : bool, default True
            If ``True``, draw a transparent ``±σ`` band around the
            per-batch curve using the within-batch standard deviation
            (per-slice scatter for slicing fits, bootstrap σ for
            whole fits, otherwise no band).
        running_mean : bool, default True
            If ``True``, overlay the cumulative running mean of the
            per-batch central tendency as a dashed line, plus a
            transparent ``±σ`` band of that cumulative series.
        title : str, optional
            Figure title. Defaults to a ``"Contact angle evolution
            ({stat})"`` string.
        save_path : str, optional
            If provided, also write the figure to standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with the per-batch line (always), the within-batch
            band (when ``per_frame_std`` and the batches expose a
            finite ``angle_std``), and the running mean line + its
            cumulative band (when ``running_mean``).
        """
        batches = list(getattr(self.results, "batches", []))
        line_color = "rgb(31, 119, 180)"
        band_fill = "rgba(31, 119, 180, 0.2)"

        if not batches:
            fig = go.Figure()
            fig.update_layout(
                title=title or f"Contact angle evolution ({self.stat})",
                xaxis_title=f"Time ({self.time_unit})",
                yaxis_title="Contact angle (°)",
                template="plotly_white",
            )
            return fig

        times = np.array([float(np.mean(b.frames)) * self.timestep for b in batches])
        per_batch = np.array([_batch_central_angle(b, self.stat) for b in batches])

        band_traces: list[go.Scatter] = []
        line_traces: list[go.Scatter] = []
        per_batch_group = self.label
        running_group = f"{self.label} running mean"

        if per_frame_std:
            std = np.array(
                [
                    float(b.angle_std)
                    if getattr(b, "angle_std", None) is not None
                    and np.isfinite(b.angle_std)
                    else float("nan")
                    for b in batches
                ]
            )
            if np.any(np.isfinite(std)):
                std_filled = np.nan_to_num(std, nan=0.0)
                band_traces.append(
                    go.Scatter(
                        x=np.concatenate([times, times[::-1]]),
                        y=np.concatenate(
                            [
                                per_batch + std_filled,
                                (per_batch - std_filled)[::-1],
                            ]
                        ),
                        fill="toself",
                        fillcolor=band_fill,
                        line={"width": 0},
                        name=f"{self.label} ±σ",
                        legendgroup=per_batch_group,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        line_traces.append(
            go.Scatter(
                x=times,
                y=per_batch,
                mode="lines",
                name=self.label,
                line={"width": 2, "color": line_color},
                legendgroup=per_batch_group,
            )
        )

        if running_mean:
            counts = np.arange(1, len(per_batch) + 1)
            cum_mean = np.cumsum(per_batch) / counts
            sq_mean = np.cumsum(per_batch**2) / counts
            cum_std = np.sqrt(np.maximum(sq_mean - cum_mean**2, 0.0))
            band_traces.append(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([cum_mean + cum_std, (cum_mean - cum_std)[::-1]]),
                    fill="toself",
                    fillcolor=band_fill,
                    line={"width": 0},
                    name=f"{self.label} running ±σ",
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
                    line={"width": 2, "color": line_color, "dash": "dash"},
                    legendgroup=running_group,
                )
            )

        fig = go.Figure()
        for trace in band_traces:
            fig.add_trace(trace)
        for trace in line_traces:
            fig.add_trace(trace)
        fig.update_layout(
            title=title or f"Contact angle evolution ({self.stat})",
            xaxis_title=f"Time ({self.time_unit})",
            yaxis_title="Contact angle (°)",
            template="plotly_white",
        )
        if save_path:
            fig.write_html(save_path)
        return fig
