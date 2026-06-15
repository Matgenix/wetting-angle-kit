"""end-to-end ``WallDetector.from_atoms`` through ``TrajectoryAnalyzer``.

Wires the ``from_atoms`` detector through the full pipeline:

1. Identify the LAMMPS substrate atom IDs (carbon, type 3 on this
   fixture).
2. Construct ``TrajectoryAnalyzer`` with ``wall_detector =
   WallDetector.from_atoms(...)`` and ``wall_atom_indices = carbon_ids``.
3. Run ``analyze`` on a single frame; verify the wall coordinates flow
   through the worker → :func:`gather_wall_coords` → :class:`WallContext`
   → ``WallDetector.detect``.

Compares against ``WallDetector.min_plus_offset(offset=0)`` on the same
fixture/frame to quantify the gap between an atom-derived wall and an
interface-derived wall.

Fixture context: the substrate is a multi-layer graphene stack with
the top layer at z ≈ 4.897 Å. That's the surface the droplet rests on
and the target of ``from_atoms(method="mean_top_layer")``.
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
def fixture_path() -> pathlib.Path:
    return _FIXTURE


@pytest.fixture
def oxygen_indices(fixture_path: pathlib.Path) -> np.ndarray:
    """LAMMPS particle IDs of the water-oxygen atoms in frame 0."""
    return LammpsDumpWaterFinder(
        fixture_path, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_indices(0)


@pytest.fixture
def carbon_indices(fixture_path: pathlib.Path) -> np.ndarray:
    """LAMMPS particle IDs of the substrate carbon atoms (type 3)."""
    # OVITO inline because the package has no general-purpose type-3-filter helper;
    # this avoids adding one just for one test.
    from ovito.io import import_file

    pipeline = cast(Any, import_file(str(fixture_path)))
    data = pipeline.compute(0)
    types = np.array(data.particles["Particle Type"].array)
    type3 = np.where(types == 3)[0]
    return np.asarray(data.particles["Particle Identifier"][type3])


def _make_analyzer(
    fixture_path: pathlib.Path,
    oxygen_indices: np.ndarray,
    wall_detector: WallDetector,
    wall_atom_indices: np.ndarray | None = None,
) -> TrajectoryAnalyzer:
    return TrajectoryAnalyzer(
        parser=LammpsDumpParser(fixture_path),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(),
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=wall_detector,
        wall_atom_indices=wall_atom_indices,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_from_atoms_wall_detector_end_to_end(
    fixture_path: pathlib.Path,
    oxygen_indices: np.ndarray,
    carbon_indices: np.ndarray,
) -> None:
    """End-to-end: WallDetector.from_atoms drives a TrajectoryAnalyzer run.

    Verifies (a) the wall coords flow through to the detector,
    (b) the analyzer produces a finite angle on the fixture, and
    (c) the recovered ``z_wall`` sits at the graphene top-layer z
    (~4.9 Å on this fixture), below the interface-derived
    ``min_plus_offset(offset=0)`` baseline (which sits a few Å higher).
    """
    frame = 1

    # 1. mean_top_layer detector — averages the top monolayer of
    #    substrate atoms.
    analyzer_atoms = _make_analyzer(
        fixture_path,
        oxygen_indices,
        wall_detector=WallDetector.from_atoms(
            wall_atom_indices=carbon_indices,
            method="mean_top_layer",
            top_layer_tolerance=1.0,
        ),
        wall_atom_indices=carbon_indices,
    )
    res_atoms = analyzer_atoms.analyze([frame])
    z_wall_atoms = float(res_atoms.batches[0].z_wall)
    angle_atoms = float(res_atoms.per_batch_angles[0])

    # 2. min_plus_offset(0) detector — interface-derived baseline.
    analyzer_min = _make_analyzer(
        fixture_path,
        oxygen_indices,
        wall_detector=WallDetector.min_plus_offset(offset=0.0),
    )
    res_min = analyzer_min.analyze([frame])
    z_wall_min = float(res_min.batches[0].z_wall)
    angle_min = float(res_min.per_batch_angles[0])

    print(
        f"\nfrom_atoms(mean_top_layer): z_wall = {z_wall_atoms:.3f} Å, "
        f"angle = {angle_atoms:.3f}°"
        f"\nmin_plus_offset(0):         z_wall = {z_wall_min:.3f} Å, "
        f"angle = {angle_min:.3f}°"
        f"\nΔz_wall = {z_wall_atoms - z_wall_min:.3f} Å, "
        f"Δangle = {angle_atoms - angle_min:.3f}°"
    )

    # Physical sanity: graphene top layer on this fixture is at z ≈ 4.9 Å.
    assert 4.5 < z_wall_atoms < 5.3, (
        f"from_atoms z_wall = {z_wall_atoms:.3f} Å; "
        f"expected ~4.9 Å for the top graphene layer."
    )
    # The interface-derived baseline sits ABOVE the wall atoms by ~1–3 Å
    # (the gap between graphene and the first liquid layer).
    assert z_wall_min > z_wall_atoms

    # Both pipelines should yield a finite, physically-plausible angle.
    assert 60.0 < angle_atoms < 130.0
    assert 60.0 < angle_min < 130.0

    # Lowering the baseline (atoms-derived vs interface-derived) raises
    # the measured angle; sign and magnitude follow
    # Δθ ≈ -Δz_wall / (R · sin θ) · (180/π) — a few degrees on this
    # fixture where R ≈ 30 Å.
    assert angle_atoms > angle_min


@pytest.mark.integration
@pytest.mark.slow
def test_from_atoms_max_z_method(
    fixture_path: pathlib.Path,
    oxygen_indices: np.ndarray,
    carbon_indices: np.ndarray,
) -> None:
    """The ``max_z`` method should land on the highest substrate atom z.

    On the fixture all top-layer carbons sit at the same z ≈ 4.897 Å,
    so ``max_z`` and ``mean_top_layer`` should agree to within
    thermal-jitter precision (< 0.1 Å).
    """
    frame = 1
    analyzer_max = _make_analyzer(
        fixture_path,
        oxygen_indices,
        wall_detector=WallDetector.from_atoms(
            wall_atom_indices=carbon_indices, method="max_z"
        ),
        wall_atom_indices=carbon_indices,
    )
    res_max = analyzer_max.analyze([frame])
    z_wall_max = float(res_max.batches[0].z_wall)

    analyzer_mean = _make_analyzer(
        fixture_path,
        oxygen_indices,
        wall_detector=WallDetector.from_atoms(
            wall_atom_indices=carbon_indices,
            method="mean_top_layer",
            top_layer_tolerance=1.0,
        ),
        wall_atom_indices=carbon_indices,
    )
    res_mean = analyzer_mean.analyze([frame])
    z_wall_mean = float(res_mean.batches[0].z_wall)

    print(
        f"\nmax_z:           z_wall = {z_wall_max:.4f} Å"
        f"\nmean_top_layer:  z_wall = {z_wall_mean:.4f} Å"
        f"\n|Δ|            = {abs(z_wall_max - z_wall_mean):.4f} Å"
    )

    # Both methods land on the same monolayer.
    assert abs(z_wall_max - z_wall_mean) < 0.1


def test_from_atoms_detector_missing_wall_coords_raises() -> None:
    """Constructing the detector without ``wall_atom_indices`` should fail loudly.

    Direct unit test on the detector (no analyzer) — verifies the
    sentinel error path that would otherwise only surface inside
    a worker.
    """
    from wetting_angle_kit.analysis.wall import WallContext, WallDetector

    detector = WallDetector.from_atoms(wall_atom_indices=np.array([1, 2, 3]))
    # Context with no wall_coords — should match the failure mode when
    # the analyzer is constructed without wall_atom_indices.
    ctx = WallContext(interface_data=np.zeros((10, 3)), wall_coords=None)
    with pytest.raises(ValueError, match="wall_coords"):
        detector.detect(ctx)
