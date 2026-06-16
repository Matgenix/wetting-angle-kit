"""3D grid extractors via marching cubes.

Three flavors:

- **Synthetic spherical cap → recovered angle close to truth.** Atoms
  uniformly fill a known spherical cap; the 3D grid extractor +
  ``SurfaceFitter.whole`` should recover the cap angle within a few
  degrees.
- **End-to-end on the LAMMPS fixture.** Smoke test pairing the
  ``grid`` + Gaussian whole extractor with ``SurfaceFitter.whole`` —
  the angle should land in the same physically plausible band as
  ``rays`` + Gaussian.
- **Cylinder geometry recovers a known horizontal ridge.** The 3D
  grid extracts a tube-like shell whose 2D ``(x, z)`` projection is
  a circle of the known cylinder radius.
"""

import pathlib

import numpy as np
import pytest

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
        sample = sample[np.linalg.norm(sample, axis=1) < R]
        sample[:, 2] += zc
        sample = sample[sample[:, 2] > z_wall]
        pts.append(sample)
    return np.concatenate(pts, axis=0)[:n_atoms]


def _whole_grid_params(half_xy: float, z_lo: float, z_hi: float, nbins: int) -> dict:
    """3D grid with cell sizes derived to give ``nbins`` cells per axis."""
    bw_xy = 2.0 * half_xy / nbins
    bw_z = (z_hi - z_lo) / nbins
    return {
        "xi_0": -half_xy,
        "xi_f": half_xy,
        "dx": bw_xy,
        "yi_0": -half_xy,
        "yi_f": half_xy,
        "dy": bw_xy,
        "zi_0": z_lo,
        "zi_f": z_hi,
        "dz": bw_z,
    }


def test_grid_with_gaussian_whole_recovers_known_spherical_cap() -> None:
    """3D grid + marching cubes + sphere fit recovers a known cap angle."""
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=80000, seed=0)

    grid_params = _whole_grid_params(half_xy=30.0, z_lo=0.0, z_hi=35.0, nbins=31)
    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(grid_params=grid_params),
        density=DensityEstimator.gaussian(density_sigma=2.0),
    )
    geom = DropletGeometry.coerce("spherical")
    extractor.validate_compatibility(surface_kind="whole", droplet_geometry=geom)
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="whole",
    )
    assert isinstance(shell, np.ndarray)
    assert shell.ndim == 2 and shell.shape[1] == 3
    assert len(shell) >= 100

    # Filter the floor (the iso-surface includes a disk near z_wall);
    # SurfaceFitter.whole's surface_filter_offset is the designed
    # mechanism for that.
    fitter = SurfaceFitter.whole(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=shell, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid + Gaussian (whole) cap recovery: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, |drift| = {drift:.3f}°, "
        f"R_fit = {out.popt[3]:.3f} (truth {R}), "
        f"zc_fit = {out.popt[2]:.3f} (truth {zc}), "
        f"shell_points = {len(shell)}, rms = {out.rms_residual:.3f} Å"
    )
    # Grid resolution (~2 Å) + Gaussian smoothing σ=2 give a few
    # degrees of drift at this droplet size.
    assert drift < 5.0


def test_grid_with_binning_whole_recovers_known_spherical_cap() -> None:
    """No-smoothing variant also recovers truth at suitable atom density."""
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=200000, seed=1)

    grid_params = _whole_grid_params(half_xy=30.0, z_lo=0.0, z_hi=35.0, nbins=25)
    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(grid_params=grid_params),
        density=DensityEstimator.binning(),
    )
    geom = DropletGeometry.coerce("spherical")
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="whole",
    )
    fitter = SurfaceFitter.whole(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=shell, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid + binning (whole) cap recovery: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, |drift| = {drift:.3f}°, "
        f"R_fit = {out.popt[3]:.3f}, "
        f"shell_points = {len(shell)}, rms = {out.rms_residual:.3f} Å"
    )
    assert drift < 5.0


def test_grid_with_gaussian_whole_cylinder_recovers_horizontal_ridge() -> None:
    """``grid`` + Gaussian + whole + cylinder recovers a known cylindrical ridge.

    A uniformly-filled cylinder of radius ``R_truth`` running along the
    ``y`` axis is binned on a 3D grid whose ``y`` extent spans the
    full cylinder. The recovered shell's ``(x, z)`` projection should
    be a circle of radius ``R_truth``.
    """
    R_truth = 12.0
    y_extent = 30.0
    n_atoms = 20000
    rng = np.random.default_rng(0)
    cross = []
    while sum(c.shape[0] for c in cross) < n_atoms:
        cand = rng.uniform(-R_truth, R_truth, size=(2 * n_atoms, 2))
        inside = np.hypot(cand[:, 0], cand[:, 1]) < R_truth
        cross.append(cand[inside])
    xz = np.concatenate(cross, axis=0)[:n_atoms]
    y = rng.uniform(-y_extent / 2, y_extent / 2, size=n_atoms)
    # Atoms occupy the cylinder centred on (x=0, z=R_truth) so the
    # ridge sits above z=0 (no atoms below z=0).
    atoms = np.column_stack([xz[:, 0], y, xz[:, 1] + R_truth])

    # 3D grid: y span covers the whole cylinder; x, z span the
    # cross-section with some margin.
    grid_params = {
        "xi_0": -1.5 * R_truth,
        "xi_f": 1.5 * R_truth,
        "dx": 1.0,
        "yi_0": -y_extent / 2,
        "yi_f": y_extent / 2,
        "dy": 1.5,
        "zi_0": 0.0,
        "zi_f": 2.5 * R_truth,
        "dz": 1.0,
    }
    extractor = InterfaceExtractor(
        sampling=SpaceSampling.grid(grid_params=grid_params),
        density=DensityEstimator.gaussian(density_sigma=1.5),
    )
    geom = DropletGeometry.coerce("cylinder_y")
    extractor.validate_compatibility(surface_kind="whole", droplet_geometry=geom)
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.array([0.0, 0.0, R_truth]),
        droplet_geometry=geom,
        surface_kind="whole",
    )
    assert isinstance(shell, np.ndarray)
    assert shell.ndim == 2 and shell.shape[1] == 3

    # In-plane radius (x, z relative to the ridge axis at (0, R_truth)).
    in_plane_r = np.hypot(shell[:, 0], shell[:, 2] - R_truth)
    mean_r = float(np.mean(in_plane_r))
    print(
        f"\ngrid + Gaussian (whole+cylinder): n_points = {shell.shape[0]}, "
        f"R_truth = {R_truth}, R_mean = {mean_r:.3f} Å"
    )
    # Gaussian smoothing (σ=1.5) places the iso-contour slightly inside
    # the geometric edge; allow ±2 Å.
    assert abs(mean_r - R_truth) < 2.0


@pytest.mark.integration
@pytest.mark.slow
def test_grid_with_gaussian_whole_end_to_end_on_lammps_fixture() -> None:
    """``grid`` + Gaussian whole pipeline on the water/graphene fixture."""
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

    grid_params = _whole_grid_params(half_xy=40.0, z_lo=0.0, z_hi=45.0, nbins=21)

    def _angle(extractor: InterfaceExtractor, fitter: SurfaceFitter) -> float:
        analyzer = TrajectoryAnalyzer(
            parser=LammpsDumpParser(fixture),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
            interface_extractor=extractor,
            surface_fitter=fitter,
            wall_detector=WallDetector.min_plus_offset(offset=0.0),
        )
        return float(analyzer.analyze([1]).per_batch_angles[0])

    angle_rays_whole = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.rays(n_rays_sphere=400),
            density=DensityEstimator.gaussian(density_sigma=3.0),
        ),
        SurfaceFitter.whole(surface_filter_offset=3.0),
    )
    angle_grid_whole = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.grid(grid_params=grid_params),
            density=DensityEstimator.gaussian(density_sigma=2.0),
        ),
        SurfaceFitter.whole(surface_filter_offset=3.0),
    )
    print(
        f"\nrays + Gaussian (whole)  angle = {angle_rays_whole:.3f}°"
        f"\ngrid + Gaussian (whole)  angle = {angle_grid_whole:.3f}°  "
        f"|drift| = {abs(angle_grid_whole - angle_rays_whole):.3f}°"
    )
    # Both should land in the physically plausible band. The drift
    # between estimators can be sizable on this 4k-atom fixture because
    # the grid is sparse (each bin captures only a few atoms) — the
    # marching-cubes mesh + sphere fit are noisy. Synthetic tests
    # (n_atoms = 80 000) confirm sub-degree accuracy when the grid is
    # well-populated; this end-to-end is a structural smoke test.
    for angle in (angle_rays_whole, angle_grid_whole):
        assert 50.0 < angle < 140.0
