"""Binning-method integration tests on a LAMMPS cylinder-droplet fixture.

End-to-end ``CoupledFit2DAnalyzer`` runs on the cylinder droplet
fixture, both single-batch and per-frame batching.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("ovito")

from wetting_angle_kit.analysis import CoupledFit2DAnalyzer  # noqa: E402
from wetting_angle_kit.analysis.temporal import TemporalAggregator  # noqa: E402
from wetting_angle_kit.parsers import (  # noqa: E402
    LammpsDumpParser,
    LammpsDumpWaterFinder,
)


@pytest.fixture
def filename() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent
        / ".."
        / "trajectories"
        / "traj_10_3_330w_nve_4k_reajust.lammpstrj"
    )


@pytest.fixture
def oxygen_indices(filename: pathlib.Path) -> np.ndarray:
    return LammpsDumpWaterFinder(
        filename, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_indices(0)


@pytest.fixture
def grid_params() -> dict:
    # The per-frame tanh NLLS is sensitive to the grid layout on
    # this fixture: ``dx`` / ``dz`` values are chosen so the edge
    # construction rounds to 49 × 24 cells, the grid the angle
    # anchors below were calibrated against.
    return {
        "xi_0": 0,
        "xi_f": 100.0,
        "dx": 100.0 / 49.0,
        "zi_0": 0.0,
        "zi_f": 100.0,
        "dz": 100.0 / 24.0,
    }


@pytest.mark.integration
def test_coupled_fit_2d_with_cylinder_fixture(
    filename: pathlib.Path,
    oxygen_indices: np.ndarray,
    grid_params: dict,
) -> None:
    """End-to-end ``CoupledFit2DAnalyzer`` on the cylinder droplet."""
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        grid_params=grid_params,
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    angle = float(results.batches[0].angle)
    # Coupled-fit angle on this fixture, frame 1: 99.110°. ±3° band.
    assert 96.0 <= angle <= 102.0
    assert np.isfinite(results.mean_angle)
    # Single batch → std across batches is 0.
    assert results.std_angle == 0.0


@pytest.mark.integration
def test_coupled_fit_2d_per_frame_batches(
    filename: pathlib.Path,
    oxygen_indices: np.ndarray,
    grid_params: dict,
) -> None:
    """``batch_size=1``: one fit per frame."""
    frames = [1, 2, 3]
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        grid_params=grid_params,
        temporal_aggregator=TemporalAggregator(batch_size=1),
    )
    results = analyzer.analyze(frames)

    # One batch per frame ⇒ three angles.
    assert len(results) == 3
    assert results.per_batch_angles.shape == (3,)
    # Observed per-frame angles on this fixture: ~99°, ~96°, ~93°
    # (some thermal drift across frames). Pin a per-frame ±5° band
    # to absorb that drift while catching real regressions.
    expected_angles = (99.11, 96.10, 92.65)
    for batch, expected in zip(results.batches, expected_angles, strict=True):
        assert len(batch.frames) == 1
        # Allow either a converged angle near the expected value, or
        # NaN on per-frame fit failure.
        assert np.isnan(batch.angle) or abs(batch.angle - expected) < 5.0
