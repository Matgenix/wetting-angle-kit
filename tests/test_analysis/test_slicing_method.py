"""Slicing-method integration test on the LAMMPS water/graphene fixture.

End-to-end ``TrajectoryAnalyzer`` with the slicing fitter + rays + Gaussian
extractor + min_plus_offset wall detector on the spherical-droplet
fixture. Anchored against the well-characterised ~95° contact angle.
"""

import pathlib

import numpy as np
import pytest

# The slicing fixture is a LAMMPS dump parsed through OVITO; skip the
# whole module when the optional dependency is unavailable (typically
# on macOS CI).
pytest.importorskip("ovito")

from wetting_angle_kit.analysis import (
    DensityEstimator,
    # noqa: E402
    InterfaceExtractor,
    SpaceSampling,
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


@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_slicing_with_real_data(
    filename: pathlib.Path, oxygen_indices: np.ndarray
) -> None:
    """End-to-end ``TrajectoryAnalyzer`` (rays + Gaussian + slicing fitter)."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(),
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    batch = results.batches[0]
    assert batch.frames == [1]
    # Slicing pipeline on this fixture: 95.159°. ±3° band.
    assert 92.0 <= batch.angle <= 98.0
    # Across-slice std on this fixture: ~1.9°.
    assert 0.5 < batch.angle_std < 4.0
    # Aggregate properties on the results container.
    assert np.isfinite(results.mean_angle)
    assert np.isfinite(results.std_angle)
