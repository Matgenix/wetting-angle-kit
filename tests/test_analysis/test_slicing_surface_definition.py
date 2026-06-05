import numpy as np
import pytest

from wetting_angle_kit.analysis.slicing.surface_definition import (
    SurfaceDefinition,
)


def _bare_surface(**overrides) -> SurfaceDefinition:
    """Build a SurfaceDefinition with defaults that test setup can override."""
    kwargs = dict(
        atom_coords=np.zeros((1, 3)),
        delta_angle=10.0,
        max_dist=20.0,
        center_geom=np.zeros(3),
        gamma=0.0,
    )
    kwargs.update(overrides)
    return SurfaceDefinition(**kwargs)


# --- density_profile (static tanh model) ---------------------------------


def test_density_profile_at_interface_equals_offset():
    # tanh(0) = 0, so rho(zd) = h regardless of d.
    z = np.array([5.0])
    rho = SurfaceDefinition.density_profile(z, zd=5.0, d=0.5, h=0.3)
    assert rho == pytest.approx(0.3)


def test_density_profile_saturates_far_from_interface():
    # tanh(+inf) = 1 (liquid side), tanh(-inf) = -1 (vapor side).
    z = np.array([-50.0, 50.0])
    rho = SurfaceDefinition.density_profile(z, zd=5.0, d=0.5, h=0.3)
    np.testing.assert_allclose(rho, [0.8, -0.2], atol=1e-10)


# --- density_contribution (Gaussian smoothing on a KD-tree) --------------


def test_density_contribution_empty_atom_set_returns_zeros():
    surf = _bare_surface(atom_coords=np.empty((0, 3)))
    positions = np.random.default_rng(0).normal(size=(7, 3))
    result = surf.density_contribution(positions)
    assert result.shape == (7,)
    np.testing.assert_array_equal(result, np.zeros(7))


def test_density_contribution_zero_samples_returns_zeros():
    surf = _bare_surface(atom_coords=np.zeros((3, 3)))
    result = surf.density_contribution(np.empty((0, 3)))
    assert result.shape == (0,)


def test_density_contribution_distant_atoms_short_circuit():
    # Single atom 173 Å from origin; 5 sigma cutoff at default sigma=3 is 15 Å.
    surf = _bare_surface(atom_coords=np.array([[100.0, 100.0, 100.0]]))
    result = surf.density_contribution(np.zeros((4, 3)))
    np.testing.assert_array_equal(result, np.zeros(4))


def test_density_contribution_peaks_at_atom_position():
    sigma = 3.0
    surf = _bare_surface(
        atom_coords=np.array([[0.0, 0.0, 0.0]]),
        density_sigma=sigma,
    )
    samples = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = surf.density_contribution(samples)
    peak = 1.0 / (2 * np.pi * sigma**2) ** 1.5
    assert result[0] == pytest.approx(peak)
    # 10 Å lies inside the 15 Å default cutoff but is heavily Gaussian-suppressed.
    expected_far = peak * np.exp(-(10.0**2) / (2 * sigma**2))
    assert result[1] == pytest.approx(expected_far)


def test_density_contribution_density_conversion_unused_in_contribution():
    # density_conversion is applied in analyze_lines, not in
    # density_contribution itself: setting it must not change this raw
    # output, which equals the bare Gaussian kernel at the sample.
    sigma = 3.0
    common = dict(
        atom_coords=np.array([[0.0, 0.0, 0.0]]),
        density_sigma=sigma,
    )
    samples = np.array([[1.0, 0.0, 0.0]])
    expected = (1.0 / (2 * np.pi * sigma**2) ** 1.5) * np.exp(-1.0 / (2 * sigma**2))
    baseline = _bare_surface(density_conversion=1.0, **common).density_contribution(
        samples
    )
    scaled = _bare_surface(density_conversion=12.5, **common).density_contribution(
        samples
    )
    assert baseline[0] == pytest.approx(expected)
    np.testing.assert_allclose(scaled, baseline)


# --- _fit_density_profiles_batched (Gauss-Newton tanh fit) ---------------


def test_fit_density_profiles_batched_recovers_known_zd():
    surf = _bare_surface(max_dist=30.0)
    z = np.linspace(0.0, 30.0, 80)
    true_zd = np.array([10.0, 15.0, 22.0])
    d, h = 0.6, 0.2
    densities = np.stack([d * np.tanh(zd - z) + h for zd in true_zd])
    fitted = surf._fit_density_profiles_batched(z, densities)
    np.testing.assert_allclose(fitted, true_zd, atol=1e-3)


def test_fit_density_profiles_batched_constant_input_falls_back_to_zero():
    # Constant density: rho_max==rho_min so d0=0 and the data midpoint
    # crossing zd0=z[argmin(0)]=z[0]=0. The first GN iteration then has a
    # singular normal matrix (j_zd = d*(1-u^2) = 0), the solver breaks,
    # and the final clip returns the seed value 0.0 exactly.
    surf = _bare_surface(max_dist=20.0)
    z = np.linspace(0.0, 20.0, 40)
    densities = np.full((2, 40), 0.5)
    fitted = surf._fit_density_profiles_batched(z, densities)
    np.testing.assert_array_equal(fitted, np.zeros(2))


# --- analyze_lines (end-to-end on a synthetic 2D droplet) ----------------


def _disk_atoms_in_xz(radius: float, n_atoms: int, seed: int) -> np.ndarray:
    """Uniform 2D disk of atoms in the y=0 slice plane."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n_atoms))
    theta = rng.uniform(0.0, 2 * np.pi, n_atoms)
    return np.column_stack([r * np.cos(theta), np.zeros(n_atoms), r * np.sin(theta)])


def test_analyze_lines_recovers_disk_radius():
    radius = 15.0
    atoms = _disk_atoms_in_xz(radius, n_atoms=4000, seed=42)
    surf = SurfaceDefinition(
        atom_coords=atoms,
        delta_angle=30.0,
        max_dist=25.0,
        center_geom=np.zeros(3),
        gamma=0.0,
        points_per_angstrom=2.0,
    )
    rr, xz = surf.analyze_lines()
    n_rays = int(360 / 30)
    assert len(rr) == n_rays
    assert len(xz) == n_rays
    assert all(len(row) == 2 for row in rr)
    assert all(len(row) == 2 for row in xz)
    # The fit pulls the apparent interface ~0.5 Å inside the geometric
    # boundary because the model uses a fixed-width tanh while the data
    # is a Gaussian-smoothed (sigma=3) step; the mismatch biases zd
    # toward the liquid side. Per-ray scatter from finite atom count is
    # ~0.3 Å on top of that.
    interface_distances = np.array([row[0] for row in rr])
    assert np.max(np.abs(interface_distances - radius)) < 1.0
    assert abs(interface_distances.mean() - radius) < 0.7


def test_analyze_lines_returns_consistent_xz_projection():
    center = np.array([5.0, 0.0, -2.0])
    atoms = _disk_atoms_in_xz(radius=10.0, n_atoms=2000, seed=0) + center
    surf = SurfaceDefinition(
        atom_coords=atoms,
        delta_angle=60.0,
        max_dist=20.0,
        center_geom=center,
        gamma=0.0,
        points_per_angstrom=2.0,
    )
    rr, xz = surf.analyze_lines()
    # Projection contract: xz[i] = center + interface_re * (cos(beta), 0, sin(beta)).
    for (re, beta), (x_proj, z_proj) in zip(rr, xz, strict=True):
        beta_rad = np.deg2rad(beta)
        assert x_proj == pytest.approx(np.cos(beta_rad) * re + center[0])
        assert z_proj == pytest.approx(np.sin(beta_rad) * re + center[2])


def test_analyze_lines_ray_count_matches_delta_angle():
    surf = _bare_surface(
        atom_coords=_disk_atoms_in_xz(radius=8.0, n_atoms=500, seed=1),
        delta_angle=45.0,
        max_dist=15.0,
    )
    rr, xz = surf.analyze_lines()
    assert len(rr) == 8
    assert len(xz) == 8
    # Each ray records its own azimuth angle in degrees, evenly spaced.
    betas = [row[1] for row in rr]
    np.testing.assert_allclose(betas, np.arange(0.0, 360.0, 45.0))
