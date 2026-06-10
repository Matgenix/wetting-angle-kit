"""Slicing-method integration tests on the LAMMPS water/graphene fixture.

Phase 11 migration: the bulk of the testing here moved to the new
``TrajectoryAnalyzer`` (slicing fitter + rays_gaussian extractor +
min_plus_offset wall detector). One legacy ``SlicingTrajectoryAnalyzer``
test is kept as a frozen regression net — scheduled for removal in
Phase 12 once :class:`SlicingTrajectoryAnalyzer` itself goes away.
"""

import pathlib

import numpy as np
import pytest

# The slicing fixture is a LAMMPS dump parsed through OVITO; skip the
# whole module when the optional dependency is unavailable (typically
# on macOS CI).
pytest.importorskip("ovito")

from wetting_angle_kit.analysis import (  # noqa: E402
    InterfaceExtractor,
    SlicingTrajectoryAnalyzer,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
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
        / "traj_spherical_drop_4k.lammpstrj"
    )


@pytest.fixture
def oxygen_indices(filename: pathlib.Path) -> np.ndarray:
    return LammpsDumpWaterFinder(
        filename, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


# --- Frozen legacy regression --------------------------------------------------
# To be removed in Phase 12 alongside ``SlicingTrajectoryAnalyzer`` itself.
@pytest.mark.integration
@pytest.mark.slow
def test_legacy_slicing_trajectory_analyzer_regression(
    filename: pathlib.Path, oxygen_indices: np.ndarray
) -> None:
    """Frozen-legacy parity check on the spherical-droplet fixture."""
    analyzer = SlicingTrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        delta_gamma=20,
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    assert results.frames == [1]
    # Legacy slicing on this fixture: mean angle = 94.873°, per-slice
    # std = 1.829°. ±3° band so the test catches real regressions
    # while tolerating numerical jitter.
    mean_angle = float(np.mean(results.angles[0]))
    assert 92.0 <= mean_angle <= 98.0
    slice_std = float(np.std(results.angles[0]))
    assert 0.5 < slice_std < 4.0


# --- New-API equivalent --------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_slicing_with_real_data(
    filename: pathlib.Path, oxygen_indices: np.ndarray
) -> None:
    """End-to-end ``TrajectoryAnalyzer`` (rays_gaussian + slicing fitter)."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor.rays_gaussian(
            delta_azimuthal=20.0, delta_polar=8.0
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    batch = results.batches[0]
    assert batch.frames == [1]
    # New slicing pipeline on this fixture: 95.159° (matches the
    # legacy 94.873° within 0.3° — see Phase 3 parity test).
    assert 92.0 <= batch.angle <= 98.0
    # Across-slice std on this fixture: ~1.9°.
    assert 0.5 < batch.angle_std < 4.0
    # Aggregate properties on the results container.
    assert np.isfinite(results.mean_angle)
    assert np.isfinite(results.std_angle)
