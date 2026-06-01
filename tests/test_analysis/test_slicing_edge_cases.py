import numpy as np
import pytest

from wetting_angle_kit.analysis.slicing.analyzer import (
    SlicingTrajectoryAnalyzer,
    _SlicingFrameResult,
)
from wetting_angle_kit.analysis.slicing.angle_fitting import (
    SlicingFrameFitter,
)


def _simple_predictor(droplet_geometry="cylinder_y", **kwargs):
    """Return a minimally-initialised SlicingFrameFitter with required attrs."""
    return SlicingFrameFitter(
        liquid_coordinates=np.zeros((10, 3)),
        max_dist=20,
        liquid_geom_center=np.array([0.0, 0.0, 0.0]),
        droplet_geometry=droplet_geometry,
        **kwargs,
    )


def test_calculate_y_axis_requires_cylinder_widths():
    predictor = _simple_predictor(
        droplet_geometry="cylinder_y", width_cylinder=10.0, delta_cylinder=2.0
    )
    # Now break them to force the validation branch in calculate_y_axis_list.
    predictor.width_cylinder = None
    with pytest.raises(ValueError, match="width_cylinder and delta_cylinder"):
        predictor.calculate_y_axis_list()


def test_calculate_gammas_requires_cylinder_widths():
    predictor = _simple_predictor(
        droplet_geometry="cylinder_y", width_cylinder=10.0, delta_cylinder=2.0
    )
    predictor.delta_cylinder = None
    with pytest.raises(ValueError, match="width_cylinder and delta_cylinder"):
        predictor.calculate_gammas_list()


def test_spherical_calculations_require_delta_gamma():
    predictor = _simple_predictor(droplet_geometry="spherical", delta_gamma=10.0)
    predictor.delta_gamma = None
    with pytest.raises(ValueError, match="delta_gamma is required"):
        predictor.calculate_y_axis_list()
    with pytest.raises(ValueError, match="delta_gamma is required"):
        predictor.calculate_gammas_list()


def test_spherical_constructor_requires_delta_gamma():
    with pytest.raises(ValueError, match="delta_gamma must be provided"):
        _simple_predictor(droplet_geometry="spherical")


def test_cylinder_constructor_raises_without_widths():
    with pytest.raises(ValueError, match="delta_cylinder and width_cylinder"):
        _simple_predictor(droplet_geometry="cylinder_y")


def test_find_intersection_returns_none_when_circle_does_not_intersect_baseline():
    predictor = _simple_predictor(
        droplet_geometry="cylinder_y", width_cylinder=10.0, delta_cylinder=2.0
    )
    # Circle center far below the baseline → no intersection
    popt = (0.0, -10.0, 1.0)
    assert predictor.find_intersection(popt, y_line=5.0) is None


def test_find_intersection_returns_angle_for_intersecting_circle():
    predictor = _simple_predictor(
        droplet_geometry="cylinder_y", width_cylinder=10.0, delta_cylinder=2.0
    )
    # Circle of radius 5 at z=0, baseline at z=0 → contact angle = 90°.
    popt = (0.0, 0.0, 5.0)
    angle = predictor.find_intersection(popt, y_line=0.0)
    assert angle == pytest.approx(90.0)


def test_calculate_y_axis_cylinder():
    predictor = _simple_predictor(
        droplet_geometry="cylinder_y", width_cylinder=10.0, delta_cylinder=2.5
    )
    assert predictor.calculate_y_axis_list() == [0.0, 2.5, 5.0, 7.5]
    assert predictor.calculate_gammas_list() == [0.0, 0.0, 0.0, 0.0]


def test_calculate_y_axis_spherical():
    predictor = _simple_predictor(
        droplet_geometry="spherical",
        delta_gamma=90.0,
    )
    # 180 / 90 = 2 entries; y_axis_list mirrors liquid_geom_center[1] each entry.
    y_axis = predictor.calculate_y_axis_list()
    gammas = predictor.calculate_gammas_list()
    assert len(y_axis) == 2
    assert len(gammas) == 2
    assert all(g >= 0 for g in gammas)


# --- SlicingTrajectoryAnalyzer worker internals ---


def test_run_one_frame_invokes_pipeline_on_real_lammps():
    """Drive ``_run_one_frame`` on a real LAMMPS fixture in the current process.

    The worker static methods normally run inside child processes, so this
    test initialises ``_WORKER_STATE`` manually and then calls
    ``_run_one_frame`` to exercise the parser → ``predict_contact_angle``
    path that subprocess execution otherwise hides from coverage.
    """
    from tests.conftest import trajectory_path

    SlicingTrajectoryAnalyzer._init_worker(
        filename=trajectory_path("traj_spherical_drop_4k.lammpstrj"),
        droplet_geometry="spherical",
        atom_indices=np.array([]),
        delta_gamma=20.0,
        delta_cylinder=None,
        points_per_angstrom=1.0,
        precentered=False,
    )
    try:
        result = SlicingTrajectoryAnalyzer._run_one_frame(0)
    finally:
        SlicingTrajectoryAnalyzer._WORKER_STATE.clear()
    assert isinstance(result, _SlicingFrameResult)
    assert result.frame_num == 0


def test_unsupported_extension_raises_at_construction(tmp_path):
    """Unknown trajectory extension must fail fast at construction, not later in
    subprocesses where the error would be silently swallowed."""
    fake = tmp_path / "trajectory.bogus"
    fake.write_text("not a real trajectory\n")

    class _FakeParser:
        filepath = str(fake)

    with pytest.raises(ValueError, match="Unsupported trajectory file format"):
        SlicingTrajectoryAnalyzer(
            parser=_FakeParser(),
            droplet_geometry="spherical",
            delta_gamma=20.0,
        )
