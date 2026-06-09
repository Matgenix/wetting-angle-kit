"""Phase 9 quantification: ``CoupledBinning2DAnalyzer`` vs legacy parity.

The new ``CoupledBinning2DAnalyzer`` is a structural rewrite of the
legacy ``BinningTrajectoryAnalyzer`` on top of the shared
``_BatchedTrajectoryAnalyzer`` scaffolding. Same per-frame projection
(``project_to_profile``), same 2D histogram + dV normalisation, same
:class:`HyperbolicTangentModel` fit. The angle should match the
legacy to floating-point precision on the same fixture.
"""

import pathlib

import numpy as np
import pytest

# Coupled binning fixtures rely on OVITO for the LAMMPS dump parser.
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

_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)


@pytest.fixture
def oxygen_indices() -> np.ndarray:
    return LammpsDumpWaterFinder(
        _FIXTURE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


def _binning_params() -> dict:
    """Explicit grid (skip the heuristic-warning code path)."""
    return {
        "xi_0": 0,
        "xi_f": 40,
        "nbins_xi": 50,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "nbins_zi": 50,
    }


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_binning_2d_matches_legacy_single_frame(
    oxygen_indices: np.ndarray,
) -> None:
    """One-frame batch: legacy and new pipelines should produce the same angle."""
    binning_params = _binning_params()

    legacy = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
    )
    legacy_results = legacy.analyze([1])
    legacy_angle = float(legacy_results.batches[0].angle)

    new = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
        temporal_aggregator=TemporalAggregator(batch_size=-1),
    )
    new_results = new.analyze([1])
    new_angle = float(new_results.batches[0].angle)

    drift = abs(legacy_angle - new_angle)
    print(
        f"\nlegacy BinningTrajectoryAnalyzer  angle = {legacy_angle:.6f}°"
        f"\nnew    CoupledBinning2DAnalyzer    angle = {new_angle:.6f}°"
        f"\n|drift|                                  = {drift:.3e}°"
    )
    # Same projection, same histogram, same fit → bit-for-bit parity.
    assert drift < 1e-9


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_binning_2d_matches_legacy_multi_frame(
    oxygen_indices: np.ndarray,
) -> None:
    """Three-frame pooled batch: same parity check across multiple frames."""
    binning_params = _binning_params()
    frames = [0, 1, 2]

    legacy = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
    )
    legacy_results = legacy.analyze(frames)
    legacy_angle = float(legacy_results.batches[0].angle)

    new = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
        temporal_aggregator=TemporalAggregator(batch_size=-1),
    )
    new_results = new.analyze(frames)
    new_angle = float(new_results.batches[0].angle)

    drift = abs(legacy_angle - new_angle)
    print(
        f"\n3-frame legacy angle = {legacy_angle:.6f}°"
        f"\n3-frame new angle    = {new_angle:.6f}°"
        f"\n|drift|              = {drift:.3e}°"
    )
    assert drift < 1e-9


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_binning_2d_split_batches_match_legacy(
    oxygen_indices: np.ndarray,
) -> None:
    """Block-pooled batches: same angles as legacy's ``split_factor`` path."""
    binning_params = _binning_params()
    frames = [0, 1, 2, 3]

    legacy = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
    )
    legacy_results = legacy.analyze(frames, split_factor=2)
    legacy_angles = sorted(float(b.angle) for b in legacy_results.batches)

    new = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params,
        temporal_aggregator=TemporalAggregator(batch_size=2),
    )
    new_results = new.analyze(frames)
    new_angles = sorted(float(b.angle) for b in new_results.batches)

    print(
        f"\nlegacy split-batch angles = {legacy_angles}"
        f"\nnew batch-size=2 angles    = {new_angles}"
    )
    assert len(legacy_angles) == len(new_angles) == 2
    for la, na in zip(legacy_angles, new_angles, strict=True):
        assert abs(la - na) < 1e-9
