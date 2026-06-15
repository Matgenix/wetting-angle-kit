"""Smoke tests for :class:`AngleEvolutionPlotter`."""

import numpy as np
import plotly.graph_objects as go
import pytest

from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit2DResults,
    SlicingBatchResult,
    TrajectoryResults,
    WholeBatchResult,
)
from wetting_angle_kit.visualization import AngleEvolutionPlotter
from wetting_angle_kit.visualization.stats import TrajectoryStats


def _slicing_results() -> TrajectoryResults:
    batches = [
        SlicingBatchResult(
            frames=[i],
            angle=95.0 + i,
            z_wall=5.0,
            rms_residual=0.1,
            angle_std=1.5,
            per_slice_angles=np.array([94.0, 95.0, 96.0]) + i,
            slice_surfaces=[np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])],
            slice_popts=np.zeros((1, 4)),
            n_slices_total=3,
            n_slices_used=3,
        )
        for i in range(3)
    ]
    return TrajectoryResults(batches=batches, method_metadata={})


def _coupled_2d_results() -> CoupledFit2DResults:
    batches = [
        CoupledFit2DBatchResult(
            frames=[i, i + 1],
            angle=99.0 - 0.5 * i,
            model_params={
                "rho1": 0.03,
                "rho2": 1e-4,
                "R_eq": 25.0,
                "zi_c": 5.0,
                "zi_0": 5.0,
                "t1": 1.0,
                "t2": 1.0,
            },
            xi_grid=np.linspace(0, 40, 10),
            zi_grid=np.linspace(0, 40, 10),
            density=np.zeros((10, 10)),
        )
        for i in range(2)
    ]
    return CoupledFit2DResults(batches=batches, method_metadata={})


def test_angle_evolution_plotter_slicing_runs_with_all_overlays() -> None:
    """Slicing results with per_frame_std + running_mean → bands + lines."""
    plotter = AngleEvolutionPlotter(
        _slicing_results(),
        label="run-A",
        timestep=2.0,
        time_unit="ps",
    )
    fig = plotter.plot(per_frame_std=True, running_mean=True)
    assert isinstance(fig, go.Figure)
    # 2 bands (within-batch + running) + 2 lines (per-batch + running mean).
    assert len(fig.data) == 4
    # x is times = frames * timestep.
    main_line = fig.data[2]
    np.testing.assert_allclose(main_line.x, [0.0, 2.0, 4.0])


def test_angle_evolution_plotter_stat_median_recomputes_central() -> None:
    """stat='median' picks median of per_slice_angles instead of batch.angle."""
    plotter = AngleEvolutionPlotter(
        _slicing_results(),
        stat="median",
    )
    fig = plotter.plot(per_frame_std=False, running_mean=False)
    # Median of [94, 95, 96] = 95 (batch 0), 96 (batch 1), 97 (batch 2).
    main_line = fig.data[0]
    np.testing.assert_allclose(main_line.y, [95.0, 96.0, 97.0])


def test_angle_evolution_plotter_stat_mean_matches_batch_angle() -> None:
    """stat='mean' matches batch.angle on slicing results."""
    plotter = AngleEvolutionPlotter(_slicing_results(), stat="mean")
    fig = plotter.plot(per_frame_std=False, running_mean=False)
    np.testing.assert_allclose(fig.data[0].y, [95.0, 96.0, 97.0])


def test_angle_evolution_plotter_coupled_results_no_band() -> None:
    """Coupled-fit batches have no angle_std → no within-batch band."""
    plotter = AngleEvolutionPlotter(_coupled_2d_results())
    fig = plotter.plot(per_frame_std=True, running_mean=False)
    # One main line, no band.
    assert len(fig.data) == 1
    np.testing.assert_allclose(fig.data[0].y, [99.0, 98.5])


def test_angle_evolution_plotter_whole_bootstrap_band() -> None:
    """``WholeBatchResult.angle_std`` from bootstrap renders the band."""
    batch = WholeBatchResult(
        frames=[0],
        angle=75.0,
        z_wall=5.0,
        rms_residual=0.1,
        angle_std=0.5,
        interface_shell=np.zeros((10, 3)),
        popt=np.array([0.0, 0.0, 0.0, 20.0, 5.0]),
    )
    results = TrajectoryResults(batches=[batch], method_metadata={})
    fig = AngleEvolutionPlotter(results).plot(per_frame_std=True, running_mean=False)
    assert isinstance(fig, go.Figure)
    # 1 band + 1 line = 2 traces.
    assert len(fig.data) == 2


def test_angle_evolution_plotter_empty_results_returns_empty_figure() -> None:
    empty = TrajectoryResults(batches=[], method_metadata={})
    fig = AngleEvolutionPlotter(empty).plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_angle_evolution_plotter_rejects_invalid_stat() -> None:
    with pytest.raises(ValueError, match="stat must be"):
        AngleEvolutionPlotter(_slicing_results(), stat="bogus")  # type: ignore[arg-type]


def test_angle_evolution_plotter_summary_returns_trajectory_stats() -> None:
    plotter = AngleEvolutionPlotter(
        _slicing_results(), label="run-A", method_name="Slicing"
    )
    summary = plotter.summary()
    assert isinstance(summary, list)
    assert len(summary) == 1
    stats = summary[0]
    assert isinstance(stats, TrajectoryStats)
    assert stats.label == "run-A"
    assert stats.method_name == "Slicing"
    assert stats.n_samples == 3
    # 1×1 unit square ⇒ shoelace area = 1.0 per frame; mean across
    # 3 frames is also 1.0.
    assert stats.mean_surface_area == pytest.approx(1.0)


def test_angle_evolution_plotter_time_axis_label() -> None:
    plotter = AngleEvolutionPlotter(_slicing_results(), timestep=0.5, time_unit="ns")
    fig = plotter.plot()
    assert fig.layout.xaxis.title.text == "Time (ns)"
