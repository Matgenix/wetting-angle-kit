"""Unit tests for the helpers behind :class:`AngleEvolutionPlotter`.

The plotter's main `.plot()` path is covered by the smoke tests in
``test_angle_evolution_plotter.py``; this file targets the internal
helpers that the smoke tests don't fully exercise — in particular
``_circular_segment_area`` over all its piecewise branches and
``_batch_surface_area`` over each result-type dispatch arm.
"""

import math

import numpy as np
import pytest

from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit3DBatchResult,
    SlicingBatchResult,
    TrajectoryResults,
    WholeBatchResult,
)
from wetting_angle_kit.visualization.angle_evolution_plotter import (
    AngleEvolutionPlotter,
    _batch_surface_area,
    _circular_segment_area,
    _shoelace_area,
)

# --- _shoelace_area ----------------------------------------------------------


def test_shoelace_empty_input_returns_zero() -> None:
    assert _shoelace_area(np.empty((0, 2))) == 0.0


def test_shoelace_unit_square() -> None:
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert _shoelace_area(sq) == pytest.approx(1.0)


# --- _circular_segment_area branches -----------------------------------------


def test_circular_segment_h_zero_or_negative_returns_zero() -> None:
    # z_center + R = z_cut → h = 0 (boundary).
    assert _circular_segment_area(R=5.0, z_center=0.0, z_cut=5.0) == 0.0
    # z_cut above the circle entirely → h < 0.
    assert _circular_segment_area(R=5.0, z_center=0.0, z_cut=10.0) == 0.0


def test_circular_segment_h_geq_2R_returns_full_circle() -> None:
    # z_cut well below the circle → segment is the whole disc.
    R = 3.0
    area = _circular_segment_area(R=R, z_center=10.0, z_cut=-100.0)
    assert area == pytest.approx(math.pi * R**2)


def test_circular_segment_small_h_branch() -> None:
    """``h <= R`` is the "less than half a disc" piecewise formula."""
    R = 1.0
    # z_center + R - z_cut = h. For h=R/2 (a small cap), check
    # against the closed-form integral.
    h = R / 2.0
    z_center = 0.0
    z_cut = z_center + R - h
    expected = R**2 * math.acos((R - h) / R) - (R - h) * math.sqrt(2 * R * h - h**2)
    assert _circular_segment_area(R, z_center, z_cut) == pytest.approx(expected)


def test_circular_segment_large_h_branch_uses_complement() -> None:
    """``R < h < 2R`` should use the "full minus small segment" branch."""
    R = 1.0
    # h = 1.5R (between R and 2R).
    h = 1.5 * R
    z_center = 0.0
    z_cut = z_center + R - h
    # Build the expected by symmetry: full circle minus the small
    # segment on the other side.
    h_small = 2 * R - h
    small_seg = R**2 * math.acos((R - h_small) / R) - (R - h_small) * math.sqrt(
        2 * R * h_small - h_small**2
    )
    expected = math.pi * R**2 - small_seg
    assert _circular_segment_area(R, z_center, z_cut) == pytest.approx(expected)


# --- _batch_surface_area dispatch --------------------------------------------


def test_batch_surface_area_slicing_uses_shoelace() -> None:
    """SlicingBatchResult: mean of per-slice shoelace areas."""
    batch = SlicingBatchResult(
        frames=[0],
        angle=90.0,
        z_wall=0.0,
        rms_residual=0.0,
        angle_std=0.0,
        per_slice_angles=np.array([90.0]),
        slice_surfaces=[
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),  # area 1
            np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),  # area 2
        ],
        slice_popts=np.zeros((2, 4)),
    )
    assert _batch_surface_area(batch) == pytest.approx(1.5)


def test_batch_surface_area_slicing_empty_surfaces_returns_zero() -> None:
    batch = SlicingBatchResult(
        frames=[0],
        angle=90.0,
        z_wall=0.0,
        rms_residual=0.0,
        angle_std=0.0,
        per_slice_angles=np.array([]),
        slice_surfaces=[],
        slice_popts=np.zeros((0, 4)),
    )
    assert _batch_surface_area(batch) == 0.0


def test_batch_surface_area_whole_spherical_popt() -> None:
    """WholeBatchResult with 5-element popt → sphere; uses zc=popt[2], R=popt[3].

    For a sphere centred at the wall (z_center = z_wall), the segment
    above the wall is the upper half-disc, area = π R² / 2.
    """
    R = 5.0
    batch = WholeBatchResult(
        frames=[0],
        angle=90.0,
        z_wall=0.0,
        rms_residual=0.0,
        angle_std=None,
        interface_shell=np.zeros((10, 3)),
        popt=np.array([0.0, 0.0, 0.0, R, 0.0]),
    )
    assert _batch_surface_area(batch) == pytest.approx(math.pi * R**2 / 2)


def test_batch_surface_area_whole_cylinder_popt() -> None:
    """4-element popt → cylinder; uses zc=popt[1], R=popt[2]."""
    R = 3.0
    batch = WholeBatchResult(
        frames=[0],
        angle=90.0,
        z_wall=0.0,
        rms_residual=0.0,
        angle_std=None,
        interface_shell=np.zeros((10, 3)),
        popt=np.array([0.0, 0.0, R, 0.0]),
    )
    assert _batch_surface_area(batch) == pytest.approx(math.pi * R**2 / 2)


def test_batch_surface_area_whole_unknown_popt_returns_nan() -> None:
    """Unexpected popt length falls through to NaN."""
    batch = WholeBatchResult(
        frames=[0],
        angle=90.0,
        z_wall=0.0,
        rms_residual=0.0,
        angle_std=None,
        interface_shell=np.zeros((10, 3)),
        popt=np.array([1.0, 2.0, 3.0]),
    )
    assert math.isnan(_batch_surface_area(batch))


def test_batch_surface_area_coupled_fit_2d_uses_model_params() -> None:
    """Both 2D and 3D coupled-binning batches share the dispatch arm."""
    batch = CoupledFit2DBatchResult(
        frames=[0],
        angle=90.0,
        model_params={
            "rho1": 0.03,
            "rho2": 1e-4,
            "R_eq": 5.0,
            "zi_c": 0.0,
            "zi_0": 0.0,
            "t1": 1.0,
            "t2": 1.0,
        },
        xi_grid=np.linspace(0, 10, 5),
        zi_grid=np.linspace(0, 10, 5),
        density=np.zeros((5, 5)),
    )
    # Same hemisphere geometry as above ⇒ area = π R² / 2.
    assert _batch_surface_area(batch) == pytest.approx(math.pi * 25.0 / 2)


def test_batch_surface_area_coupled_fit_3d_uses_model_params() -> None:
    batch = CoupledFit3DBatchResult(
        frames=[0],
        angle=90.0,
        model_params={
            "rho1": 0.03,
            "rho2": 1e-4,
            "R_eq": 5.0,
            "xi_c": 0.0,
            "yi_c": 0.0,
            "zi_c": 0.0,
            "zi_0": 0.0,
            "t1": 1.0,
            "t2": 1.0,
        },
        xi_grid=np.linspace(-10, 10, 5),
        yi_grid=np.linspace(-10, 10, 5),
        zi_grid=np.linspace(0, 10, 5),
        density=np.zeros((5, 5, 5)),
    )
    assert _batch_surface_area(batch) == pytest.approx(math.pi * 25.0 / 2)


def test_batch_surface_area_unknown_type_returns_nan() -> None:
    """Anything not in the dispatch table falls through to NaN."""
    assert math.isnan(_batch_surface_area("not a batch"))


# --- summary() integration ---------------------------------------------------


def test_summary_empty_results_returns_nan_stats() -> None:
    plotter = AngleEvolutionPlotter(
        TrajectoryResults(batches=[], method_metadata={}),
        label="empty-traj",
    )
    summary = plotter.summary()
    assert len(summary) == 1
    stats = summary[0]
    assert stats.label == "empty-traj"
    assert stats.n_samples == 0
    assert math.isnan(stats.mean_surface_area)
    assert math.isnan(stats.mean_contact_angle)
