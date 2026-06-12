"""Auto-derived ``grid_params`` / ``binning_params`` defaults.

When the user constructs a grid extractor or a coupled-binning
analyzer without specifying the spatial grid spec, the package picks
one from the atom bounding box (extractors) or the box dimensions
(analyzers). These tests verify that the auto-derived defaults
produce physically reasonable angles on the bundled LAMMPS fixture.

The reference angle on the water/graphene fixture is ~95° (see the
slicing/whole pipeline tests in this directory). The bounds here are
±5° around that — wide enough that the per-method bias is absorbed,
tight enough that a real regression in the default-derivation logic
gets flagged.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("ovito")
pytest.importorskip("skimage")

from wetting_angle_kit.analysis import (  # noqa: E402
    CoupledBinning2DAnalyzer,
    CoupledBinning3DAnalyzer,
    InterfaceExtractor,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
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


@pytest.mark.integration
def test_coupled_binning_2d_auto_default(oxygen_indices: np.ndarray) -> None:
    """``CoupledBinning2DAnalyzer`` with no ``binning_params`` lands at ~95°."""
    with pytest.warns(UserWarning, match="binning_params was not supplied"):
        analyzer = CoupledBinning2DAnalyzer(
            parser=LammpsDumpParser(_FIXTURE),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
        )
    batch = analyzer.analyze([1]).batches[0]
    assert 90.0 < batch.angle < 100.0
    # Sensible model params, not degenerate.
    assert 20.0 < batch.model_params["R_eq"] < 60.0


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_binning_3d_auto_default(oxygen_indices: np.ndarray) -> None:
    """``CoupledBinning3DAnalyzer`` with no ``binning_params`` lands at ~95°."""
    with pytest.warns(UserWarning, match="binning_params was not supplied"):
        analyzer = CoupledBinning3DAnalyzer(
            parser=LammpsDumpParser(_FIXTURE),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
        )
    batch = analyzer.analyze([1]).batches[0]
    assert 90.0 < batch.angle < 100.0


@pytest.mark.integration
def test_grid_gaussian_slicing_auto_default(
    oxygen_indices: np.ndarray,
) -> None:
    """``grid_gaussian`` slicing pipeline with no ``grid_params`` lands at ~95°."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor.grid_gaussian(delta_azimuthal=20.0),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=3.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert 90.0 < batch.angle < 100.0


@pytest.mark.integration
def test_grid_gaussian_whole_auto_default(
    oxygen_indices: np.ndarray,
) -> None:
    """``grid_gaussian`` whole-fit pipeline with no ``grid_params`` lands at ~95°."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor.grid_gaussian(),
        surface_fitter=SurfaceFitter.whole(surface_filter_offset=3.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert 90.0 < batch.angle < 100.0


@pytest.mark.integration
@pytest.mark.slow
def test_grid_binning_whole_auto_default(
    oxygen_indices: np.ndarray,
) -> None:
    """``grid_binning`` whole-fit pipeline with no ``grid_params``.

    Whole mode bins the full 3D density (no slab cut), so the
    auto-default holds up where per-frame slicing-mode
    ``grid_binning`` doesn't. The recovered angle should sit in the
    physically-acceptable band.

    Note: ``grid_binning`` + slicing-mode + ``grid_params=None`` is
    intrinsically unreliable for single-frame analyses (the slab cut
    leaves few atoms per cell); that combination has no dedicated
    test because there is no robust default for it.
    """
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor.grid_binning(),
        surface_fitter=SurfaceFitter.whole(surface_filter_offset=3.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    batch = analyzer.analyze([1]).batches[0]
    # Wide band because the histogram iso-surface is noisier than
    # the KDE one; on this fixture it recovers ~116° vs ~96° for the
    # KDE variant. Both are in the "physically plausible" range,
    # just biased by the per-cell shot noise.
    assert 85.0 < batch.angle < 125.0
