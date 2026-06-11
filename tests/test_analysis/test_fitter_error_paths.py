"""Direct unit tests for :class:`SurfaceFitter` error paths.

The :class:`TrajectoryAnalyzer` integration tests cover the happy
path through both fitter kinds; this file targets the explicit
``raise`` branches that the integration fixtures don't hit.
"""

import numpy as np
import pytest

from wetting_angle_kit.analysis import SurfaceFitter
from wetting_angle_kit.analysis.geometry import DropletGeometry


@pytest.fixture
def spherical() -> DropletGeometry:
    return DropletGeometry.coerce("spherical")


@pytest.fixture
def cylinder() -> DropletGeometry:
    return DropletGeometry.coerce("cylinder_y")


# --- Slicing fitter -----------------------------------------------------------


def test_slicing_rejects_non_list_input(spherical: DropletGeometry) -> None:
    fitter = SurfaceFitter.slicing()
    with pytest.raises(TypeError, match="list of per-slice"):
        fitter.fit(np.zeros((10, 3)), z_wall=0.0, droplet_geometry=spherical)


def test_slicing_skips_empty_and_thin_slices(spherical: DropletGeometry) -> None:
    """Slices with zero / too few points after filtering must be skipped."""
    fitter = SurfaceFitter.slicing(surface_filter_offset=2.0)
    # A valid spherical-cap arc to anchor the batch — without it the
    # fitter raises "no valid slice", which is a different code path.
    theta = np.linspace(np.pi * 0.55, np.pi - 0.1, 50)
    R = 20.0
    valid_slice = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    surfaces = [
        np.empty((0, 2)),  # empty slice → skipped
        np.array([[0.0, 10.0], [1.0, 10.5]]),  # only 2 points → skipped
        valid_slice,
    ]
    out = fitter.fit(surfaces, z_wall=0.0, droplet_geometry=spherical)
    # Only the valid slice contributed.
    assert len(out.per_slice_angles) == 1
    assert len(out.slice_surfaces) == 1


def test_slicing_skips_circle_outside_wall(spherical: DropletGeometry) -> None:
    """Circles whose centre is too far from the wall (|Δz| ≥ R) are skipped."""
    fitter = SurfaceFitter.slicing(surface_filter_offset=0.0)
    # A circle that sits high above the wall (centre at z=50, R=5).
    # |z_wall - z_c| = 50 > R = 5 → no intersection.
    theta = np.linspace(np.pi * 0.1, np.pi * 0.9, 40)
    high_slice = np.column_stack([5.0 * np.cos(theta), 50.0 + 5.0 * np.sin(theta)])
    # And a valid spherical-cap arc that intersects z=0.
    theta = np.linspace(np.pi * 0.55, np.pi - 0.1, 50)
    R = 20.0
    valid_slice = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    out = fitter.fit([high_slice, valid_slice], z_wall=0.0, droplet_geometry=spherical)
    # Only the valid slice produces an angle.
    assert len(out.per_slice_angles) == 1


def test_slicing_raises_when_no_valid_slice(spherical: DropletGeometry) -> None:
    """If every slice is dropped, the fitter raises rather than averaging zero."""
    fitter = SurfaceFitter.slicing(surface_filter_offset=2.0)
    # Every slice below the filter → all skipped.
    bad_surfaces = [
        np.array([[0.0, -5.0], [1.0, -4.5]]),
        np.array([[0.0, -3.0], [1.0, -2.5]]),
    ]
    with pytest.raises(RuntimeError, match="no slice produced"):
        fitter.fit(bad_surfaces, z_wall=0.0, droplet_geometry=spherical)


# --- Whole fitter -------------------------------------------------------------


def test_whole_rejects_negative_bootstrap() -> None:
    with pytest.raises(ValueError, match="bootstrap_samples must be"):
        SurfaceFitter.whole(bootstrap_samples=-1)


def test_whole_rejects_non_ndarray_input(spherical: DropletGeometry) -> None:
    fitter = SurfaceFitter.whole()
    with pytest.raises(TypeError, match="\\(N, 3\\) ndarray shell"):
        fitter.fit([np.zeros((4, 2))], z_wall=0.0, droplet_geometry=spherical)


def test_whole_rejects_wrong_shell_shape(spherical: DropletGeometry) -> None:
    fitter = SurfaceFitter.whole()
    with pytest.raises(ValueError, match="\\(N, 3\\) ndarray shell"):
        fitter.fit(np.zeros((10, 2)), z_wall=0.0, droplet_geometry=spherical)


def test_whole_rejects_insufficient_points(spherical: DropletGeometry) -> None:
    """Below the geometric-fit minimum, the fitter raises a clear RuntimeError."""
    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    # Only two points above z_wall=0; need 4 for a sphere fit.
    shell = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, -1.0],  # below filter
            [1.0, 3.0, -1.0],  # below filter
        ]
    )
    with pytest.raises(RuntimeError, match="sphere fit"):
        fitter.fit(shell, z_wall=0.0, droplet_geometry=spherical)


def test_whole_raises_when_cap_does_not_intersect_wall(
    spherical: DropletGeometry,
) -> None:
    """A sphere far from the wall (|Δz| ≥ R) yields no contact angle."""
    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    # A sphere centred at (0, 0, 100) with R ≈ 1. Wall at z=0 → no intersect.
    n_phi, n_theta = 12, 6
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.05, np.pi - 0.05, n_theta)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    R_sphere = 1.0
    shell = np.column_stack(
        [
            (R_sphere * np.sin(T) * np.cos(P)).ravel(),
            (R_sphere * np.sin(T) * np.sin(P)).ravel(),
            100.0 + (R_sphere * np.cos(T)).ravel(),
        ]
    )
    with pytest.raises(RuntimeError, match="does not intersect"):
        fitter.fit(shell, z_wall=0.0, droplet_geometry=spherical)


def test_whole_cylinder_geometry_uses_circle_fit(
    cylinder: DropletGeometry,
) -> None:
    """Cylinder droplet → 2D circle fit; popt has 4 entries (xc, zc, R, z_wall)."""
    fitter = SurfaceFitter.whole(surface_filter_offset=0.0)
    # A cylinder of radius 10 along y, centred at (x=0, z=8) so it
    # crosses the wall at z=0 (|Δz|=8 < R=10 → finite contact angle).
    n_phi = 100
    n_y = 20
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ys = np.linspace(-10.0, 10.0, n_y)
    R_cyl = 10.0
    z_center = 8.0
    points = []
    for y in ys:
        x = R_cyl * np.cos(phi)
        z = z_center + R_cyl * np.sin(phi)
        points.append(np.column_stack([x, np.full(n_phi, y), z]))
    shell = np.concatenate(points, axis=0)
    out = fitter.fit(shell, z_wall=0.0, droplet_geometry=cylinder)
    # popt = [xc, zc, R, z_wall] for cylinder; the centre is at (0, 8).
    assert out.popt.shape == (4,)
    np.testing.assert_allclose(out.popt[0], 0.0, atol=0.1)
    np.testing.assert_allclose(out.popt[1], z_center, atol=0.1)
    np.testing.assert_allclose(out.popt[2], R_cyl, atol=0.1)
    # cos θ = (0 - 8) / 10 = -0.8 → θ = arccos(-0.8) ≈ 143.13°.
    expected = float(np.degrees(np.arccos(-0.8)))
    np.testing.assert_allclose(out.angle, expected, atol=1.0)


def test_whole_bootstrap_populates_angle_std(
    spherical: DropletGeometry,
) -> None:
    """``bootstrap_samples=20`` returns a finite angle_std."""
    fitter = SurfaceFitter.whole(surface_filter_offset=0.0, bootstrap_samples=20)
    # A noisy partial sphere with cap intersecting z=0.
    rng = np.random.default_rng(0)
    n_phi, n_theta = 40, 12
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.05, np.pi * 0.6, n_theta)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    R_sphere = 10.0
    noise = rng.normal(0, 0.1, size=(n_phi * n_theta, 3))
    shell = (
        np.column_stack(
            [
                (R_sphere * np.sin(T) * np.cos(P)).ravel(),
                (R_sphere * np.sin(T) * np.sin(P)).ravel(),
                (R_sphere * np.cos(T)).ravel(),
            ]
        )
        + noise
    )
    out = fitter.fit(shell, z_wall=0.0, droplet_geometry=spherical)
    assert out.angle_std is not None
    assert 0.0 < out.angle_std < 5.0
