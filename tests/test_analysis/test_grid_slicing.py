"""Grid extractors (slicing mode).

Three flavors:

- **Synthetic spherical droplet → recovered angle ≈ truth.** Atoms
  inside a spherical cap of known truth angle; the grid extractor +
  slicing fitter pipeline should recover the angle within a few
  degrees.
- **grid + binning vs grid + Gaussian on the same atoms.** Both should
  recover similar interfaces; the KDE-smoothed variant produces
  cleaner (lower per-slice RMS) contours than the bare histogram.
- **End-to-end on the LAMMPS water/graphene fixture.** Grid extractors
  paired with the slicing fitter should produce angles within a few
  degrees of ``rays`` + Gaussian on the same fixture/frame.

All grid-mode slicing extractors take ``delta_azimuthal`` (spherical)
or ``delta_cylinder`` (cylinder) and iterate per-slice — the
returned contour list has one entry per azimuthal slice for
spherical, one per y-step for cylinder.
"""

import pathlib

import numpy as np
import pytest

# Skip entirely if scikit-image isn't installed — both grid extractors
# depend on it.
pytest.importorskip("skimage")

from wetting_angle_kit.analysis import (
    DensityEstimator,
    # noqa: E402
    InterfaceExtractor,
    SpaceSampling,
    SurfaceFitter,
    WallDetector,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry  # noqa: E402


def _spherical_cap_atoms(
    *,
    R: float,
    zc: float,
    z_wall: float,
    n_atoms: int,
    seed: int = 0,
) -> np.ndarray:
    """Atoms uniformly filling a spherical cap above ``z_wall``."""
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while sum(p.shape[0] for p in pts) < n_atoms:
        sample = rng.uniform(-R, R, size=(4 * n_atoms, 3))
        inside_sphere = np.linalg.norm(sample, axis=1) < R
        sample = sample[inside_sphere]
        sample[:, 2] += zc
        sample = sample[sample[:, 2] > z_wall]
        pts.append(sample)
    return np.concatenate(pts, axis=0)[:n_atoms]


def _default_grid_params(half: float) -> dict[str, object]:
    """A symmetric ``(s, z)`` slice grid: ``s ∈ [-half, half]``, ``z ∈ [0, half]``."""
    return {
        "xi_0": -half,
        "xi_f": half,
        "dx": half / 25.0,
        "zi_0": 0.0,
        "zi_f": half,
        "dz": half / 25.0,
    }


def test_grid_with_gaussian_recovers_known_spherical_cap_angle() -> None:
    """Per-azimuthal-slice ``grid`` + Gaussian recovers a known cap angle."""
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=15000, seed=0)

    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(
            grid_params=_default_grid_params(half=35.0), delta_azimuthal=30.0
        ),
        density=DensityEstimator.gaussian(density_sigma=2.0),
    )
    geom = DropletGeometry.coerce("spherical")
    extractor.validate_compatibility(surface_kind="slicing", droplet_geometry=geom)
    contours = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )
    assert isinstance(contours, list)
    # delta_azimuthal=30° → 6 slices.
    assert len(contours) == 6
    for contour in contours:
        assert contour.ndim == 2 and contour.shape[1] == 2
        assert len(contour) >= 10

    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=contours, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid + Gaussian (6 slices) cap recovery: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, |drift| = {drift:.3f}°, "
        f"per_slice σ = {out.angle_std:.3f}°, "
        f"rms_residual = {out.rms_residual:.3f} Å"
    )
    assert drift < 2.0
    # Axisymmetric truth ⇒ per-slice scatter should be sub-degree.
    assert out.angle_std < 2.0


def test_grid_with_binning_recovers_known_spherical_cap_with_coarse_bins() -> None:
    """``grid`` + binning (no smoothing) needs coarser cells to give a usable contour.

    On finer grids the Poisson noise per bin dominates and the slab
    cut becomes thin; the coarse-cells case shows the tool produces
    sensible answers with the right configuration.
    """
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=50000, seed=0)

    # Per-slice binning sees only the atoms within the slab, so the
    # per-cell count is low. A 4 Å in-plane bin + 4 Å slab thickness
    # keeps that count high enough that Poisson noise doesn't dominate
    # the iso-contour at the 95th-percentile bulk estimator.
    grid_params: dict[str, object] = {
        "xi_0": -35.0,
        "xi_f": 35.0,
        "dx": 4.0,
        "zi_0": 0.0,
        "zi_f": 35.0,
        "dz": 2.0,
    }
    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(
            grid_params=grid_params,
            delta_azimuthal=60.0,  # 3 slices
        ),
        density=DensityEstimator.binning(),
    )
    geom = DropletGeometry.coerce("spherical")
    contours = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )
    assert len(contours) == 3

    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=contours, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid + binning (coarse, 3 slices) cap recovery: "
        f"truth = {truth_angle:.3f}°, recovered = {out.angle:.3f}°, "
        f"|drift| = {drift:.3f}°, rms_residual = {out.rms_residual:.3f} Å"
    )
    assert drift < 5.0


def test_grid_with_gaussian_smoother_than_grid_with_binning() -> None:
    """At equal grid spec, ``grid`` + Gaussian gives a smoother contour."""
    R, zc, z_wall = 25.0, 0.0, 5.0
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=50000, seed=1)

    # Coarse grid with thick slab so grid + binning isn't dominated by
    # Poisson noise; same spec for both estimators so the comparison
    # is apples-to-apples.
    grid_params: dict[str, object] = {
        "xi_0": -35.0,
        "xi_f": 35.0,
        "dx": 4.0,
        "zi_0": 0.0,
        "zi_f": 35.0,
        "dz": 2.0,
    }
    geom = DropletGeometry.coerce("spherical")

    b = InterfaceExtractor(
        sampling=SpaceSampling.grid(grid_params=grid_params, delta_azimuthal=60.0),
        density=DensityEstimator.binning(),
    )
    g = InterfaceExtractor(
        sampling=SpaceSampling.grid(grid_params=grid_params, delta_azimuthal=60.0),
        density=DensityEstimator.gaussian(density_sigma=2.0),
    )

    binning_contours = b.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )
    gaussian_contours = g.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )

    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out_b = fitter.fit(
        interface_data=binning_contours, z_wall=z_wall, droplet_geometry=geom
    )
    out_g = fitter.fit(
        interface_data=gaussian_contours, z_wall=z_wall, droplet_geometry=geom
    )
    print(
        f"\ngrid + binning  rms = {out_b.rms_residual:.3f} Å, "
        f"angle = {out_b.angle:.3f}°"
        f"\ngrid + Gaussian rms = {out_g.rms_residual:.3f} Å, "
        f"angle = {out_g.angle:.3f}°"
    )
    assert out_g.rms_residual <= out_b.rms_residual
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    assert abs(out_b.angle - truth_angle) < 8.0
    assert abs(out_g.angle - truth_angle) < 5.0


def test_grid_with_gaussian_rejects_missing_delta_azimuthal_for_spherical() -> None:
    """slicing+spherical with ``delta_azimuthal=None`` must fail at validation."""
    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(
            grid_params=_default_grid_params(half=35.0), delta_azimuthal=None
        ),
        density=DensityEstimator.gaussian(density_sigma=2.0),
    )
    with pytest.raises(ValueError, match="delta_azimuthal"):
        extractor.validate_compatibility(
            surface_kind="slicing",
            droplet_geometry=DropletGeometry.coerce("spherical"),
        )


@pytest.mark.integration
@pytest.mark.slow
def test_grid_extractors_end_to_end_close_to_rays_with_gaussian() -> None:
    """Grid extractor angles on the LAMMPS fixture sit within a few ° of rays."""
    pytest.importorskip("ovito")
    from wetting_angle_kit.analysis import TrajectoryAnalyzer
    from wetting_angle_kit.parsers import (
        LammpsDumpParser,
        LammpsDumpWaterFinder,
    )

    fixture = (
        pathlib.Path(__file__).parent
        / ".."
        / "trajectories"
        / "traj_spherical_drop_4k.lammpstrj"
    )
    finder = LammpsDumpWaterFinder(fixture, oxygen_type=1, hydrogen_type=2)
    oxygen_indices = finder.get_water_oxygen_indices(0)

    grid_params_gauss = {
        "xi_0": -40.0,
        "xi_f": 40.0,
        "dx": 3.0,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "dz": 1.6,
    }
    # grid + binning per-slice has fewer atoms per cell than rays + binning
    # (only atoms in the slab contribute, not all atoms along a ray):
    # need a thick slab AND few slices to keep per-bin counts reasonable.
    grid_params_bin = {
        "xi_0": -40.0,
        "xi_f": 40.0,
        "dx": 8.0,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "dz": 3.0,
    }

    def _angle(extractor: InterfaceExtractor) -> float:
        analyzer = TrajectoryAnalyzer(
            parser=LammpsDumpParser(fixture),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
            interface_extractor=extractor,
            surface_fitter=SurfaceFitter.slicing(surface_filter_offset=3.0),
            wall_detector=WallDetector.min_plus_offset(offset=0.0),
        )
        return float(analyzer.analyze([1]).per_batch_angles[0])

    angle_rays = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        )
    )
    angle_grid_bin = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.grid(
                grid_params=grid_params_bin, delta_azimuthal=60.0
            ),
            density=DensityEstimator.binning(),
        )
    )
    angle_grid_gauss = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.grid(
                grid_params=grid_params_gauss, delta_azimuthal=20.0
            ),
            density=DensityEstimator.gaussian(density_sigma=2.0),
        )
    )
    print(
        f"\nrays + Gaussian               angle = {angle_rays:.3f}°"
        f"\ngrid + binning (slab=5 Å)     angle = {angle_grid_bin:.3f}°  "
        f"|drift| = {abs(angle_grid_bin - angle_rays):.3f}°"
        f"\ngrid + Gaussian (slab=3 Å σ)  angle = {angle_grid_gauss:.3f}°  "
        f"|drift| = {abs(angle_grid_gauss - angle_rays):.3f}°"
    )
    for angle in (angle_rays, angle_grid_bin, angle_grid_gauss):
        assert 70.0 < angle < 110.0
    assert abs(angle_grid_gauss - angle_rays) < 8.0
    assert abs(angle_grid_bin - angle_rays) < 14.0
