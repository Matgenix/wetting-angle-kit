"""Synthetic-correctness tests for ``SurfaceFitter.slicing()``.

Per-slice points lying on a known circle. The fitter should recover
the truth contact angle to numerical precision and report (near-)zero
fit residual.
"""

import numpy as np

from wetting_angle_kit.analysis import SurfaceFitter
from wetting_angle_kit.analysis.geometry import DropletGeometry

# --- Synthetic correctness -------------------------------------------------


def _circle_arc_points(
    xc: float, zc: float, radius: float, theta_lo: float, theta_hi: float, n: int
) -> np.ndarray:
    """Sample (x, z) points along an arc of a circle."""
    theta = np.linspace(theta_lo, theta_hi, n)
    return np.column_stack([xc + radius * np.cos(theta), zc + radius * np.sin(theta)])


def test_slicing_fitter_recovers_known_circle_angle() -> None:
    """Per-slice points on a known circle → recovered angle matches truth."""
    # Circle of radius R centred at (0, zc); wall at z=0. The cap
    # extends from the contact line up to the top of the circle.
    # cos θ = (z_wall - zc) / R, so for zc = -5 and R = 20:
    # cos θ = 0.25  →  θ ≈ 75.522°.
    xc_truth, zc_truth, radius_truth = 0.0, -5.0, 20.0
    z_wall = 0.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / radius_truth)))

    # Sample the upper arc of the circle (z > z_wall).
    # theta in radians measured from +x axis; upper arc is [0, pi].
    arc = _circle_arc_points(
        xc=xc_truth,
        zc=zc_truth,
        radius=radius_truth,
        theta_lo=np.arcsin((z_wall - zc_truth) / radius_truth),
        theta_hi=np.pi - np.arcsin((z_wall - zc_truth) / radius_truth),
        n=80,
    )
    # ``surface_filter_offset = 0`` so every arc point is kept.
    fitter = SurfaceFitter.slicing(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=[arc],
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("spherical"),
    )

    print(
        f"\nSynthetic circle: truth = {truth_angle:.4f}°, "
        f"recovered = {out.angle:.4f}°, rms_residual = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 1e-6
    # Algebraic Taubin on exact-circle points fits to ~floating-point
    # precision; the per-slice RMS residual should sit near 0.
    assert out.rms_residual < 1e-9
    # One slice → angle_std = 0.
    assert out.angle_std == 0.0


def test_slicing_fitter_aggregates_across_slices() -> None:
    """Multiple noisy slices around the same true angle — aggregation."""
    radius_truth = 20.0
    zc_truth = -5.0
    z_wall = 0.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / radius_truth)))

    rng = np.random.default_rng(7)
    slices: list[np.ndarray] = []
    for _ in range(8):
        arc = _circle_arc_points(
            xc=0.0,
            zc=zc_truth,
            radius=radius_truth,
            theta_lo=np.arcsin((z_wall - zc_truth) / radius_truth),
            theta_hi=np.pi - np.arcsin((z_wall - zc_truth) / radius_truth),
            n=40,
        )
        # ±0.05 Å Gaussian noise on both axes — sub-resolution thermal jitter.
        arc = arc + rng.normal(0.0, 0.05, size=arc.shape)
        slices.append(arc)

    fitter = SurfaceFitter.slicing(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=slices,
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("spherical"),
    )

    print(
        f"Aggregated 8 noisy slices: truth = {truth_angle:.3f}°, "
        f"mean = {out.angle:.3f}°, angle_std = {out.angle_std:.3f}°, "
        f"rms_residual = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 0.5
    # Per-slice std should be small and finite.
    assert out.angle_std > 0.0
    assert out.angle_std < 1.0
    assert out.per_slice_angles.shape == (8,)
    assert out.slice_popts.shape == (8, 4)
    assert len(out.slice_surfaces) == 8
