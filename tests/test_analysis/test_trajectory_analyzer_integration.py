"""End-to-end integration coverage for :class:`TrajectoryAnalyzer`.

Covers the configurations not exercised by the per-component tests:

- ``TrajectoryAnalyzer`` + whole-fit + ``rays`` + Gaussian on real data;
- slicing pipeline on a cylinder LAMMPS fixture;
- bootstrap σ_θ flowing through the analyzer into ``WholeBatchResult``;
- multi-frame pooled batches via ``TemporalAggregator(batch_size=N)``;
- ``WallDetector.from_atoms`` paired with the whole fitter.
"""

import pathlib
from typing import Any, cast

import numpy as np
import pytest

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
from wetting_angle_kit.analysis.results import (  # noqa: E402
    SlicingBatchResult,
    WholeBatchResult,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator  # noqa: E402
from wetting_angle_kit.parsers import (  # noqa: E402
    LammpsDumpParser,
    LammpsDumpWaterFinder,
)

_SPHERICAL_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)
_CYLINDER_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_10_3_330w_nve_4k_reajust.lammpstrj"
)


@pytest.fixture
def spherical_oxygen_ids() -> np.ndarray:
    return LammpsDumpWaterFinder(
        _SPHERICAL_FIXTURE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


@pytest.fixture
def cylinder_oxygen_ids() -> np.ndarray:
    return LammpsDumpWaterFinder(
        _CYLINDER_FIXTURE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


@pytest.fixture
def spherical_carbon_ids() -> np.ndarray:
    """Carbon (type 3) particle IDs on the spherical fixture."""
    from ovito.io import import_file

    pipeline = cast(Any, import_file(str(_SPHERICAL_FIXTURE)))
    data = pipeline.compute(0)
    types = np.array(data.particles["Particle Type"].array)
    type3 = np.where(types == 3)[0]
    return np.asarray(data.particles["Particle Identifier"][type3])


# --- whole-fit pipeline end-to-end --------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_whole_fit_rays_with_gaussian_on_spherical(
    spherical_oxygen_ids: np.ndarray,
) -> None:
    """End-to-end ``TrajectoryAnalyzer`` with ``SurfaceFitter.whole()``."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(n_rays_sphere=400),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        surface_fitter=SurfaceFitter.whole(surface_filter_offset=3.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    results = analyzer.analyze([1])

    assert len(results) == 1
    batch = results.batches[0]
    assert isinstance(batch, WholeBatchResult)
    # Full-sphere Fibonacci rays from the COM hit the wall on the way
    # down → ``min(shell z) ≈ z_wall`` → ``min_plus_offset(0)``
    # returns the physical wall position → the whole-fit recovers
    # the physical contact angle (~95.4°) matching the slicing /
    # binning pipelines on this fixture. ±3° band.
    assert 92.0 < batch.angle < 99.0
    # WholeBatchResult fields: shell, popt of length 5 (xc, yc, zc, R, z_wall).
    assert batch.interface_shell.ndim == 2 and batch.interface_shell.shape[1] == 3
    assert batch.popt.shape == (5,)
    # No bootstrap configured ⇒ angle_std is None.
    assert batch.angle_std is None
    assert batch.rms_residual > 0.0
    # RMS residual on this fixture sits ~1.3 Å; flag growth past 3 Å.
    assert batch.rms_residual < 3.0
    # ``z_wall`` is the interface-derived baseline; observed ~6.5 Å.
    assert 5.5 < batch.z_wall < 8.0


@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_whole_fit_with_explicit_wall(
    spherical_oxygen_ids: np.ndarray,
) -> None:
    """Whole-fit at the physical wall position (graphene top ~4.9 Å).

    With the full-sphere Fibonacci, ``min_plus_offset`` already
    finds the physical wall; setting it explicitly is mostly a
    convenience for trajectories where the wall position is known a
    priori from the simulation setup. On this fixture the explicit
    wall is ~1.6 Å below the interface-derived baseline, lifting the
    measured angle from ~95° to ~99° (Δθ ≈ -Δz_wall / (R sin θ)).
    """
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(n_rays_sphere=400),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        surface_fitter=SurfaceFitter.whole(surface_filter_offset=3.0),
        wall_detector=WallDetector.explicit(z_wall=4.9),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert isinstance(batch, WholeBatchResult)
    # Observed 99.12° on this fixture; ±2° band.
    assert 97.0 < batch.angle < 101.0
    # ``z_wall`` is reported back as supplied.
    assert batch.z_wall == pytest.approx(4.9)


@pytest.mark.integration
@pytest.mark.slow
def test_whole_fitter_bootstrap_through_analyzer(
    spherical_oxygen_ids: np.ndarray,
) -> None:
    """``bootstrap_samples > 0`` populates ``WholeBatchResult.angle_std``."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(n_rays_sphere=400),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        surface_fitter=SurfaceFitter.whole(
            surface_filter_offset=3.0, bootstrap_samples=100
        ),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert isinstance(batch, WholeBatchResult)
    assert batch.angle_std is not None
    assert batch.angle_std > 0.0
    # The 400-ray shell on this fixture is well-determined; observed
    # σ_θ ~ 0.5°. Any value above 2° points at a numerical regression.
    assert batch.angle_std < 2.0
    # Same pipeline as the companion test above ⇒ ~95° band.
    assert 92.0 < batch.angle < 99.0


@pytest.mark.integration
@pytest.mark.slow
def test_whole_fit_with_from_atoms_wall_detector(
    spherical_oxygen_ids: np.ndarray, spherical_carbon_ids: np.ndarray
) -> None:
    """``WallDetector.from_atoms`` flowing into the whole fitter end-to-end."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(n_rays_sphere=400),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        surface_fitter=SurfaceFitter.whole(surface_filter_offset=3.0),
        wall_detector=WallDetector.from_atoms(
            wall_atom_indices=spherical_carbon_ids,
            method="mean_top_layer",
            top_layer_tolerance=1.0,
        ),
        wall_atom_indices=spherical_carbon_ids,
    )
    batch = analyzer.analyze([1]).batches[0]
    assert isinstance(batch, WholeBatchResult)
    # Top graphene layer on this fixture sits at z ≈ 4.897 Å.
    assert 4.5 < batch.z_wall < 5.3
    # The from_atoms wall sits ~1.6 Å below the interface-derived
    # baseline → angle ~ 99° here (vs 95° with min_plus_offset(0)).
    # ±2° around the observed value.
    assert 97.0 < batch.angle < 101.0


# --- slicing pipeline on cylinder data ----------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_slicing_pipeline_on_cylinder_lammps_fixture(
    cylinder_oxygen_ids: np.ndarray,
) -> None:
    """New ``TrajectoryAnalyzer`` slicing fit on the cylinder LAMMPS fixture."""
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_CYLINDER_FIXTURE),
        atom_indices=cylinder_oxygen_ids,
        droplet_geometry="cylinder_y",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_cylinder=4.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert isinstance(batch, SlicingBatchResult)
    # On this cylinder fixture the slicing pipeline recovers ~102°;
    # ±4° accommodates per-slice scatter.
    assert 98.0 < batch.angle < 106.0
    # Six slices fall out of ``y_extent / delta_cylinder``; per-slice
    # scatter on this fixture is sub-3°.
    assert batch.per_slice_angles.shape == (6,)
    assert 0.5 < batch.angle_std < 4.0


# --- multi-frame pooled batching ----------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_pooled_batches(
    spherical_oxygen_ids: np.ndarray,
) -> None:
    """``TemporalAggregator(batch_size=2)`` produces 2-frame pooled batches."""
    frames = [0, 1, 2, 3]
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(),
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
        temporal_aggregator=TemporalAggregator(batch_size=2),
    )
    results = analyzer.analyze(frames)
    assert len(results) == 2
    for batch in results.batches:
        # Each batch pools two consecutive frames.
        assert len(batch.frames) == 2
        # Pooled batches sit within ±3° of the per-frame angle (~95°)
        # on this fixture; observed 94.8°.
        assert 91.0 < batch.angle < 98.0


@pytest.mark.integration
@pytest.mark.slow
def test_trajectory_analyzer_fully_pooled(
    spherical_oxygen_ids: np.ndarray,
) -> None:
    """``batch_size=-1`` pools every requested frame into a single fit."""
    frames = [0, 1, 2, 3]
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(_SPHERICAL_FIXTURE),
        atom_indices=spherical_oxygen_ids,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(),
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
        temporal_aggregator=TemporalAggregator(batch_size=-1),
    )
    results = analyzer.analyze(frames)
    assert len(results) == 1
    batch = results.batches[0]
    assert batch.frames == frames
    # Fully pooled angle on this fixture: 94.5°; ±3.5° band.
    assert 91.0 < batch.angle < 98.0
