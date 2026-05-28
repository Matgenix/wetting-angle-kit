import os

import numpy as np
import pytest

from wetting_angle_kit.analysis.slicing.angle_fitting import (
    ContactAngleSlicing,
)
from wetting_angle_kit.analysis.slicing.parallel import (
    ContactAngleSlicingParallel,
    SlicingFrameResult,
)


def _simple_predictor(droplet_geometry="cylinder_y", **kwargs):
    """Return a minimally-initialised ContactAngleSlicing with required attrs."""
    return ContactAngleSlicing(
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


def test_cylinder_constructor_warns_without_widths():
    with pytest.warns(UserWarning, match="recommended"):
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


# --- ContactAngleSlicingParallel internals ---


def test_create_batches_few_frames(tmp_path):
    parallel = ContactAngleSlicingParallel(filename="ignored", output_dir=str(tmp_path))
    # num_batches >= len(frames) → one frame per batch
    assert parallel._create_batches([1, 2, 3], num_batches=4) == [[1], [2], [3]]


def test_create_batches_many_frames(tmp_path):
    parallel = ContactAngleSlicingParallel(filename="ignored", output_dir=str(tmp_path))
    batches = parallel._create_batches(list(range(10)), num_batches=3)
    flat = [f for batch in batches for f in batch]
    assert flat == list(range(10))
    # Approximately equal split; each batch is ≤ ceil(10/3) = 4.
    assert all(len(b) <= 4 for b in batches)


def test_process_batch_worker_invokes_pipeline_on_real_lammps(tmp_path):
    """Run _process_batch_worker on a real LAMMPS fixture in the current process.

    Goes through detect_parser_type → LammpsDumpParser → predict_contact_angle,
    so it exercises the worker code paths that subprocess execution otherwise
    hides from coverage.
    """
    from tests.conftest import trajectory_path

    parallel = ContactAngleSlicingParallel(
        filename=trajectory_path("traj_spherical_drop_4k.lammpstrj"),
        output_dir=str(tmp_path),
        droplet_geometry="spherical",
        delta_gamma=20.0,
    )
    results = parallel._process_batch_worker(batch_frames=[0])
    assert len(results) == 1
    assert isinstance(results[0], SlicingFrameResult)
    assert results[0].frame_num == 0


def test_process_batch_worker_unsupported_extension(tmp_path):
    """Unknown trajectory extension → worker returns failed results."""
    fake = tmp_path / "trajectory.bogus"
    fake.write_text("not a real trajectory\n")
    parallel = ContactAngleSlicingParallel(
        filename=str(fake),
        output_dir=str(tmp_path),
        droplet_geometry="spherical",
        delta_gamma=20.0,
    )
    results = parallel._process_batch_worker(batch_frames=[0, 1])
    assert len(results) == 2
    assert all(r.mean_angle is None for r in results)


def test_output_dir_is_created(tmp_path):
    target = tmp_path / "nested" / "out"
    parallel = ContactAngleSlicingParallel(filename="ignored", output_dir=str(target))
    assert os.path.isdir(target)
    assert parallel.output_dir == str(target)
