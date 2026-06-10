"""Quantification tests for the ``rays_gaussian`` extractor.

- **Fibonacci correctness** for the whole+spherical case: build a
  synthetic uniform-volume sphere of atoms and verify the recovered
  shell sits near the sphere radius and spans both hemispheres.
- **Cylinder ridge smoke test** for the whole+cylinder case.
"""

import numpy as np
import pytest

from wetting_angle_kit.analysis.extractors import InterfaceExtractor
from wetting_angle_kit.analysis.geometry import DropletGeometry


def _uniform_sphere_atoms(radius: float, n_atoms: int, seed: int = 0) -> np.ndarray:
    """Rejection-sample atoms uniformly inside a sphere of the given radius."""
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    target = n_atoms
    while sum(len(p) for p in pts) < target:
        chunk = rng.uniform(-radius, radius, size=(target * 3, 3))
        mask = np.linalg.norm(chunk, axis=1) <= radius
        pts.append(chunk[mask])
    return np.concatenate(pts, axis=0)[:target]


def test_whole_spherical_recovers_known_sphere_radius() -> None:
    """Fibonacci sampling on the whole hemisphere recovers a known sphere.

    Builds a uniform-volume sphere of atoms and runs the new
    whole+spherical extractor with rays emitted from the sphere centre.
    Each shell point should sit at the sphere radius (modulo a small
    inward shift from the Gaussian density smoothing).
    """
    radius = 20.0
    sigma = 3.0
    # Use a full sphere of atoms (no hemisphere cut) so the
    # Fibonacci-spaced full-sphere rays probe an angularly isotropic
    # atom cloud. This isolates the sampling pattern + tanh-fit
    # recovery from any cap-induced bias near the equator.
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
    # Full-sphere Fibonacci directions span ``cos θ ∈ [-1, 1]``; the
    # recovered shell should cover both hemispheres roughly equally
    # — no zero crossing in ``z`` is allowed to be biased.
    assert np.any(shell[:, 2] < 0)
    assert np.any(shell[:, 2] > 0)
    # Symmetric cloud → mean z should sit near zero.
    assert abs(float(np.mean(shell[:, 2]))) < 1.0

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
