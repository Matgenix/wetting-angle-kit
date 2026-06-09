"""Quantification tests for the new ``rays_gaussian`` extractor.

Two flavors:

- **Parity** with the legacy slicing primitive
  (:class:`SurfaceDefinition.analyze_lines`) on slicing-mode cases.
  Both pipelines route through the same
  :mod:`wetting_angle_kit.analysis._density` helpers, so agreement
  should be bit-for-bit (max-abs diff at the 1e-12 level).
- **Fibonacci correctness** for the new whole+spherical case where
  there's no legacy comparator. We build a synthetic uniform-volume
  sphere of atoms and verify the recovered shell sits at the sphere
  radius to within the density-smoothing tolerance.
"""

import numpy as np
import pytest

from wetting_angle_kit.analysis._density import (
    DEFAULT_CUTOFF_SIGMA,
    DEFAULT_DENSITY_SIGMA,
)
from wetting_angle_kit.analysis.extractors import InterfaceExtractor
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.slicing.surface_definition import (
    SurfaceDefinition,
)


def _make_random_droplet(
    seed: int = 0, n_atoms: int = 2000, sigma_xy: float = 8.0, sigma_z: float = 6.0
) -> np.ndarray:
    """Random atom cloud that loosely resembles a sessile droplet."""
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.0, sigma_xy, size=(n_atoms, 2))
    z = np.abs(rng.normal(0.0, sigma_z, size=n_atoms)) + 1.0
    return np.column_stack([xy, z])


def test_slicing_spherical_matches_legacy_surface_definition() -> None:
    """Parity: new extractor (slicing+spherical) vs legacy SurfaceDefinition."""
    atoms = _make_random_droplet(seed=0)
    center = np.array([0.0, 0.0, 0.0])
    delta_azimuthal = 30.0
    delta_polar = 8.0
    max_dist = 30.0

    # Legacy: instantiate one SurfaceDefinition per gamma and call
    # analyze_lines() — exactly what the legacy
    # SlicingFrameFitter does for spherical droplets.
    n_slices = int(180 / delta_azimuthal)
    gammas = np.linspace(0.0, 180.0, n_slices)
    legacy_xz: list[np.ndarray] = []
    for gamma in gammas:
        sd = SurfaceDefinition(
            atom_coords=atoms,
            delta_angle=delta_polar,
            max_dist=max_dist,
            center_geom=center,
            gamma=float(gamma),
            density_sigma=DEFAULT_DENSITY_SIGMA,
            cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
        )
        _, xz = sd.analyze_lines()
        legacy_xz.append(np.asarray(xz, dtype=float))

    # New: rays_gaussian extractor in slicing+spherical mode.
    extractor = InterfaceExtractor.rays_gaussian(
        delta_azimuthal=delta_azimuthal,
        delta_polar=delta_polar,
        density_sigma=DEFAULT_DENSITY_SIGMA,
        cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
    )
    geom = DropletGeometry.coerce("spherical")
    extractor.validate_compatibility(surface_kind="slicing", droplet_geometry=geom)
    new_xz = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=center,
        droplet_geometry=geom,
        max_dist=max_dist,
        surface_kind="slicing",
    )

    assert isinstance(new_xz, list)
    assert len(new_xz) == len(legacy_xz)

    max_diff = 0.0
    for legacy, new in zip(legacy_xz, new_xz, strict=True):
        assert legacy.shape == new.shape
        diff = float(np.max(np.abs(legacy - new)))
        max_diff = max(max_diff, diff)

    # Quantification: agreement is at the floating-point round-off
    # level (the two paths call the same density / tanh helpers).
    print(f"\nslicing+spherical parity: max |diff| = {max_diff:.3e} Å")
    assert max_diff < 1e-10


def test_slicing_cylinder_matches_legacy_surface_definition() -> None:
    """Parity check for the cylinder slicing case."""
    atoms = _make_random_droplet(seed=1)
    center = np.array([0.0, 0.0, 0.0])
    delta_cylinder = 4.0
    delta_polar = 8.0
    max_dist = 30.0

    # Legacy: gamma stays at 0, y sweeps over delta_cylinder steps.
    y_vals = atoms[:, 1]
    ys = np.arange(float(y_vals.min()), float(y_vals.max()), delta_cylinder)
    legacy_xz: list[np.ndarray] = []
    for y in ys:
        slice_center = np.array([center[0], float(y), center[2]])
        sd = SurfaceDefinition(
            atom_coords=atoms,
            delta_angle=delta_polar,
            max_dist=max_dist,
            center_geom=slice_center,
            gamma=0.0,
            density_sigma=DEFAULT_DENSITY_SIGMA,
            cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
        )
        _, xz = sd.analyze_lines()
        legacy_xz.append(np.asarray(xz, dtype=float))

    extractor = InterfaceExtractor.rays_gaussian(
        delta_cylinder=delta_cylinder,
        delta_polar=delta_polar,
        density_sigma=DEFAULT_DENSITY_SIGMA,
        cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
    )
    geom = DropletGeometry.coerce("cylinder_y")
    extractor.validate_compatibility(surface_kind="slicing", droplet_geometry=geom)
    new_xz = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=center,
        droplet_geometry=geom,
        max_dist=max_dist,
        surface_kind="slicing",
    )

    assert isinstance(new_xz, list)
    assert len(new_xz) == len(legacy_xz)
    max_diff = 0.0
    for legacy, new in zip(legacy_xz, new_xz, strict=True):
        assert legacy.shape == new.shape
        diff = float(np.max(np.abs(legacy - new)))
        max_diff = max(max_diff, diff)
    print(f"slicing+cylinder parity: max |diff| = {max_diff:.3e} Å")
    assert max_diff < 1e-10


def _uniform_sphere_atoms(radius: float, n_atoms: int, seed: int = 0) -> np.ndarray:
    """Atoms uniformly distributed inside a sphere of given radius."""
    rng = np.random.default_rng(seed)
    # Rejection-sample inside a cube; cheap and unbiased.
    pts = []
    while len(pts) * 3 < n_atoms * 3:
        sample = rng.uniform(-radius, radius, size=(4 * n_atoms, 3))
        inside = np.linalg.norm(sample, axis=1) < radius
        pts.append(sample[inside])
        if sum(len(p) for p in pts) >= n_atoms:
            break
    return np.concatenate(pts, axis=0)[:n_atoms]


def test_whole_spherical_recovers_known_sphere_radius() -> None:
    """Fibonacci sampling on the whole hemisphere recovers a known sphere.

    Builds a uniform-volume sphere of atoms and runs the new
    whole+spherical extractor with rays emitted from the sphere centre.
    Each shell point should sit at the sphere radius (modulo a small
    inward shift from the Gaussian density smoothing).
    """
    radius = 20.0
    sigma = 3.0
    # Use a full sphere (no hemisphere cut) so the Fibonacci-spaced
    # upper-hemisphere rays probe an angularly isotropic atom cloud.
    # This isolates the sampling pattern + tanh-fit recovery from any
    # cap-induced inward bias near the equator.
    atoms = _uniform_sphere_atoms(radius=radius, n_atoms=15000, seed=0)

    n_rays = 400
    extractor = InterfaceExtractor.rays_gaussian(
        n_rays_sphere=n_rays,
        density_sigma=sigma,
    )
    geom = DropletGeometry.coerce("spherical")
    extractor.validate_compatibility(surface_kind="whole", droplet_geometry=geom)
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=radius + 10.0,
        surface_kind="whole",
    )
    assert isinstance(shell, np.ndarray)
    assert shell.shape == (n_rays, 3)
    # Fibonacci hemisphere directions all have cos θ ≥ 0; the
    # recovered shell points should mirror that.
    assert np.all(shell[:, 2] >= -1e-12)

    r = np.linalg.norm(shell, axis=1)
    mean_r = float(np.mean(r))
    std_r = float(np.std(r))
    max_dev = float(np.max(np.abs(r - radius)))

    print(
        "\nFibonacci sphere recovery: "
        f"R_truth = {radius} Å, R_mean = {mean_r:.3f} Å, "
        f"R_std = {std_r:.3f} Å, max |R_i - R| = {max_dev:.3f} Å"
    )

    # Tolerance: density smoothing with sigma=3 places the tanh-fit
    # interface at the half-max density, which can drift up to ~sigma
    # from the geometric edge. Mean radius should land within sigma.
    assert abs(mean_r - radius) < sigma
    # The angular spread should be much smaller than the smoothing.
    assert std_r < 1.0
    # No outliers far from the truth radius.
    assert max_dev < 1.5 * sigma


@pytest.mark.unit
def test_whole_cylinder_recovers_horizontal_ridge() -> None:
    """Smoke test: whole+cylinder recovers a horizontal cylindrical ridge.

    A uniformly-filled cylinder of radius R extending along y; recovered
    shell points (x, *, z) should land on a circle of radius R in the
    (x, z) plane at every sampled y.
    """
    R_truth = 15.0
    y_extent = 30.0
    n_atoms = 8000
    rng = np.random.default_rng(2)
    # Uniform within radius R in (x, z); uniform along y.
    cross = []
    while sum(c.shape[0] for c in cross) < n_atoms:
        cand = rng.uniform(-R_truth, R_truth, size=(2 * n_atoms, 2))
        inside = np.hypot(cand[:, 0], cand[:, 1]) < R_truth
        cross.append(cand[inside])
    xz = np.concatenate(cross, axis=0)[:n_atoms]
    y = rng.uniform(-y_extent / 2, y_extent / 2, size=n_atoms)
    atoms = np.column_stack([xz[:, 0], y, xz[:, 1] + R_truth])
    # Shift so atoms sit above z = 0 to mimic the sessile-droplet frame.

    extractor = InterfaceExtractor.rays_gaussian(
        delta_cylinder=3.0,
        delta_polar=8.0,
    )
    geom = DropletGeometry.coerce("cylinder_y")
    extractor.validate_compatibility(surface_kind="whole", droplet_geometry=geom)
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.array([0.0, 0.0, R_truth]),
        max_dist=R_truth + 10.0,
        droplet_geometry=geom,
        surface_kind="whole",
    )
    assert isinstance(shell, np.ndarray)
    assert shell.ndim == 2 and shell.shape[1] == 3

    # In-plane radius (x, z relative to the centre we passed in).
    in_plane_r = np.hypot(shell[:, 0], shell[:, 2] - R_truth)
    mean_r = float(np.mean(in_plane_r))
    print(
        f"whole+cylinder shell: n_points = {shell.shape[0]}, "
        f"R_truth = {R_truth}, R_mean = {mean_r:.3f} Å"
    )
    assert abs(mean_r - R_truth) < 1.5  # density-smoothing tolerance
