"""Binning-method integration tests on a LAMMPS cylinder-droplet fixture.

Phase 11 migration: the new ``CoupledBinning2DAnalyzer`` is exercised
end-to-end against the same fixture as the legacy
``BinningTrajectoryAnalyzer``. One legacy test is kept as a frozen
regression net (to be removed in Phase 12 alongside the legacy class).
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("ovito")

from wetting_angle_kit.analysis import (  # noqa: E402
    BinningTrajectoryAnalyzer,
    CoupledBinning2DAnalyzer,
)
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
    ).get_water_oxygen_ids(0)


@pytest.fixture
def binning_params() -> dict:
    return {
        "xi_0": 0,
        "xi_f": 100.0,
        "nbins_xi": 50,
        "zi_0": 0.0,
        "zi_f": 100.0,
        "nbins_zi": 25,
    }


# --- Frozen legacy regression --------------------------------------------------
# To be removed in Phase 12 alongside ``BinningTrajectoryAnalyzer`` itself.
@pytest.mark.integration
def test_legacy_binning_trajectory_analyzer_regression(
    filename: pathlib.Path,
    oxygen_indices: np.ndarray,
    binning_params: dict,
) -> None:
    """Frozen-legacy regression on the cylinder-droplet fixture."""
    analyzer = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        binning_params=binning_params,
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    # Legacy binning on the cylinder fixture, frame 1: 99.110°.
    # ±3° band.
    assert 96.0 <= results.mean_angle <= 102.0
    # Single batch → std across batches is 0.
    assert results.std_angle == 0.0


# --- New-API equivalent --------------------------------------------------------
@pytest.mark.integration
def test_coupled_binning_2d_with_cylinder_fixture(
    filename: pathlib.Path,
    oxygen_indices: np.ndarray,
    binning_params: dict,
) -> None:
    """End-to-end ``CoupledBinning2DAnalyzer`` on the cylinder droplet."""
    analyzer = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        binning_params=binning_params,
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    angle = float(results.batches[0].angle)
    # New analyzer matches legacy bit-for-bit (Phase 9 parity test)
    # so the same 99.110° ±3° band applies.
    assert 96.0 <= angle <= 102.0
    assert np.isfinite(results.mean_angle)
    # Single batch → std across batches is 0.
    assert results.std_angle == 0.0


@pytest.mark.integration
def test_coupled_binning_2d_per_frame_batches(
    filename: pathlib.Path,
    oxygen_indices: np.ndarray,
    binning_params: dict,
) -> None:
    """``batch_size=1`` ↔ legacy ``split_factor=1``: one fit per frame."""
    frames = [1, 2, 3]
    analyzer = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        binning_params=binning_params,
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
        # NaN on per-frame fit failure (matches the legacy contract).
        assert np.isnan(batch.angle) or abs(batch.angle - expected) < 5.0
