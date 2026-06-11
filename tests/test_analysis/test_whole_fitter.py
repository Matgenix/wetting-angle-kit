"""Phase 4 quantification: ``SurfaceFitter.whole()`` correctness + bootstrap.

Four flavors:

- **Exact-sphere recovery.** Feed the fitter exact Fibonacci-sphere
  shell points; verify the recovered angle matches truth to numerical
  precision and the RMS residual sits near zero.
- **Exact-cylinder recovery.** Same for a straight cylinder along ``y``.
- **End-to-end with the rays_gaussian extractor.** Synthetic atom sphere
  → extractor → fitter; angle should track truth within the
  density-smoothing budget.
- **Bootstrap σ scaling.** On a noisy shell, the bootstrap σ_θ should
  scale like ``1/√N_shell``. Quantified for three shell sizes.
"""

import numpy as np

from wetting_angle_kit.analysis.extractors import InterfaceExtractor
from wetting_angle_kit.analysis.extractors._sampling import (
    _fibonacci_sphere_directions,
)
from wetting_angle_kit.analysis.fitters import SurfaceFitter
from wetting_angle_kit.analysis.geometry import DropletGeometry


def _uniform_sphere_atoms(radius: float, n_atoms: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while sum(p.shape[0] for p in pts) < n_atoms:
        sample = rng.uniform(-radius, radius, size=(4 * n_atoms, 3))
        pts.append(sample[np.linalg.norm(sample, axis=1) < radius])
    return np.concatenate(pts, axis=0)[:n_atoms]


def test_whole_fitter_exact_sphere_recovers_angle_to_numerical_precision() -> None:
    """Exact Fibonacci-sphere shell → angle within < 1e-3°."""
    R_truth = 20.0
    zc_truth = 0.0
    z_wall = 5.0  # cos θ = 0.25 → θ ≈ 75.522°
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / R_truth)))

    directions = _fibonacci_sphere_directions(400)
    shell = directions * R_truth + np.array([0.0, 0.0, zc_truth])

    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=shell,
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("spherical"),
    )

    print(
        f"\nExact sphere: truth = {truth_angle:.6f}°, "
        f"recovered = {out.angle:.6f}°, "
        f"R = {out.popt[3]:.6f}, zc = {out.popt[2]:.6f}, "
        f"rms = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 1e-3
    assert out.rms_residual < 1e-9
    assert out.angle_std is None  # bootstrap disabled


def test_whole_fitter_exact_cylinder_recovers_angle_to_numerical_precision() -> None:
    """Exact cylinder shell → angle within < 1e-3°."""
    R_truth = 15.0
    zc_truth = 0.0
    z_wall = 4.0  # cos θ = 4/15 → θ ≈ 74.474°
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / R_truth)))

    # Half-cylinder along y, radius R_truth, in (x, z) plane.
    polar = np.linspace(0, 360, 45, endpoint=False)
    cos_polar = np.cos(np.deg2rad(polar))
    sin_polar = np.sin(np.deg2rad(polar))
    y_vals = np.arange(-15.0, 15.0, 3.0)
    shell_parts: list[np.ndarray] = []
    for y in y_vals:
        shell_parts.append(
            np.column_stack(
                [
                    R_truth * cos_polar,
                    np.full_like(polar, y),
                    R_truth * sin_polar + zc_truth,
                ]
            )
        )
    shell = np.concatenate(shell_parts, axis=0)

    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=shell,
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("cylinder_y"),
    )

    print(
        f"\nExact cylinder: truth = {truth_angle:.6f}°, "
        f"recovered = {out.angle:.6f}°, "
        f"R = {out.popt[2]:.6f}, zc = {out.popt[1]:.6f}, "
        f"rms = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 1e-3
    assert out.rms_residual < 1e-9
    # popt for cylinder: [xc, zc, R, z_wall]
    assert out.popt.shape == (4,)


def test_whole_fitter_end_to_end_atom_sphere() -> None:
    """Full pipeline (extractor → fitter) on a synthetic atom sphere.

    The recovered angle has a density-smoothing bias: the tanh fit
    locates the interface at the half-max density, which sits slightly
    inside the geometric edge. That shifts R inward (~0.7 Å for σ=3)
    and shifts the recovered angle a few degrees.
    """
    R_truth = 20.0
    z_wall = 5.0
    truth_angle = float(np.degrees(np.arccos(z_wall / R_truth)))
    atoms = _uniform_sphere_atoms(radius=R_truth, n_atoms=15000, seed=0)

    extractor = InterfaceExtractor.rays_gaussian(n_rays_sphere=400, density_sigma=3.0)
    geom = DropletGeometry.coerce("spherical")
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=R_truth + 10.0,
        surface_kind="whole",
    )

    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=shell,
        z_wall=z_wall,
        droplet_geometry=geom,
    )

    print(
        f"\nEnd-to-end atom sphere: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, "
        f"R_recovered = {out.popt[3]:.3f} (truth {R_truth}), "
        f"zc = {out.popt[2]:.3f} (truth 0.0)"
    )
    # Two compounding smoothing biases (at σ_density=3):
    #   1. The tanh fit locates the interface at the half-density
    #      contour, which sits ~0.7 Å inside the geometric edge → R
    #      shrinks from 20 to ~19.7.
    #   2. The hemisphere-weighted Fibonacci sampling combined with
    #      the density evaluator at the centre pulls the fitted zc
    #      slightly downward (~ -0.8 Å on this fixture).
    # The net effect on the angle is a 2.6° drift at this scale; the
    # bias would shrink with smaller σ_density or larger R.
    assert abs(out.angle - truth_angle) < 3.0


def test_whole_fitter_bootstrap_sigma_scales_inverse_sqrt_n_shell() -> None:
    """Bootstrap σ_θ on a noisy shell scales like ``1/√N_shell``.

    With shell sizes (200, 800, 3200) — a 4× ratio between adjacent
    levels — the σ ratio should land near 2 (one factor of √4).
    """
    R_truth = 20.0
    zc_truth = 0.0
    z_wall = 5.0
    point_noise = 0.5  # Å Gaussian noise per shell point

    def make_noisy_shell(n_rays: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        directions = _fibonacci_sphere_directions(n_rays)
        shell = directions * R_truth + np.array([0.0, 0.0, zc_truth])
        return shell + rng.normal(0.0, point_noise, size=shell.shape)

    geom = DropletGeometry.coerce("spherical")
    sigmas: dict[int, float] = {}
    for n_rays in (200, 800, 3200):
        shell = make_noisy_shell(n_rays, seed=42)
        fitter = SurfaceFitter.whole(surface_filter_offset=0.0, bootstrap_samples=300)
        out = fitter.fit(interface_data=shell, z_wall=z_wall, droplet_geometry=geom)
        assert out.angle_std is not None
        sigmas[n_rays] = out.angle_std

    print(
        "\nBootstrap σ_θ vs shell size (Å noise = 0.5):"
        + "".join(f"\n  N = {n:5d}: σ_θ = {s:.4f}°" for n, s in sigmas.items())
    )

    ratio_200_800 = sigmas[200] / sigmas[800]
    ratio_800_3200 = sigmas[800] / sigmas[3200]
    ratio_200_3200 = sigmas[200] / sigmas[3200]
    print(
        f"  σ(200)  / σ(800)  = {ratio_200_800:.3f}  (expected √4 ≈ 2.00)\n"
        f"  σ(800)  / σ(3200) = {ratio_800_3200:.3f}  (expected √4 ≈ 2.00)\n"
        f"  σ(200)  / σ(3200) = {ratio_200_3200:.3f}  (expected √16 ≈ 4.00)"
    )

    # Each 4× step should give a σ ratio near 2; total 16× near 4.
    # Allow ±35% slack because the bootstrap estimator's own variance
    # at 300 resamples isn't negligible.
    assert 1.4 < ratio_200_800 < 2.7
    assert 1.4 < ratio_800_3200 < 2.7
    assert 2.7 < ratio_200_3200 < 5.4
