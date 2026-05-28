"""Unit tests for the shared (r, z) projection used by
:class:`BaseParser.get_profile_coordinates`.

These exercise all three droplet geometries (including ``cylinder_x``
which was not previously covered by any test).
"""

import numpy as np
import pytest

from wetting_angle_kit.io_utils import project_to_profile


def _grid_points(n=20):
    """Return a centered grid of 3D points: cube from -5..5, z from 0..10."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    z = rng.uniform(0.0, 10.0, size=(n, 1))
    return np.hstack([xy, z])


def test_project_cylinder_y_uses_centered_x_as_radial():
    pts = _grid_points()
    r, z = project_to_profile(pts, "cylinder_y")
    # Radial coord must be |x_centered|.
    x_cm = pts.mean(axis=0)
    expected_r = np.abs(pts[:, 0] - x_cm[0])
    np.testing.assert_allclose(r, expected_r)
    # z stays in lab frame (no centering).
    np.testing.assert_allclose(z, pts[:, 2])
    assert (r >= 0).all()


def test_project_cylinder_x_uses_centered_y_as_radial():
    pts = _grid_points()
    r, z = project_to_profile(pts, "cylinder_x")
    y_cm = pts.mean(axis=0)[1]
    expected_r = np.abs(pts[:, 1] - y_cm)
    np.testing.assert_allclose(r, expected_r)
    np.testing.assert_allclose(z, pts[:, 2])
    assert (r >= 0).all()


def test_project_spherical_uses_centered_radial_distance():
    pts = _grid_points()
    r, z = project_to_profile(pts, "spherical")
    cm = pts.mean(axis=0)
    expected_r = np.sqrt((pts[:, 0] - cm[0]) ** 2 + (pts[:, 1] - cm[1]) ** 2)
    np.testing.assert_allclose(r, expected_r)
    np.testing.assert_allclose(z, pts[:, 2])
    assert (r >= 0).all()


def test_project_cylinder_x_and_cylinder_y_are_axis_swaps():
    """Swapping x and y should map cylinder_y onto cylinder_x."""
    pts = _grid_points()
    r_y, _ = project_to_profile(pts, "cylinder_y")
    swapped = pts[:, [1, 0, 2]]
    r_x, _ = project_to_profile(swapped, "cylinder_x")
    np.testing.assert_allclose(r_y, r_x)


def test_project_empty_input_returns_empty_arrays():
    empty = np.empty((0, 3))
    r, z = project_to_profile(empty, "spherical")
    assert r.shape == (0,)
    assert z.shape == (0,)


def test_project_rejects_unknown_geometry():
    pts = _grid_points(3)
    with pytest.raises(ValueError, match="Unknown droplet_geometry"):
        project_to_profile(pts, "blob")


def _localized_cluster(box_length: float, n: int = 30) -> np.ndarray:
    """Return n points clustered tightly enough around the box center that
    the circular mean and the arithmetic mean agree to numerical precision
    (the difference scales like (spread/L)^2, so a 1% spread is plenty)."""
    rng = np.random.default_rng(0)
    center = box_length / 2.0
    spread = box_length / 100.0
    xy = rng.uniform(center - spread, center + spread, size=(n, 2))
    z = rng.uniform(0.0, box_length / 2.0, size=(n, 1))
    return np.hstack([xy, z])


def test_project_with_box_size_matches_legacy_for_mid_box_cluster():
    """When the droplet sits comfortably away from the boundary, the
    PBC-aware path returns the same radii as the legacy arithmetic-mean
    path: minimum-image folding is a no-op and the circular mean coincides
    with the arithmetic mean."""
    box = (20.0, 20.0)
    pts = _localized_cluster(box[0])
    r_legacy, z_legacy = project_to_profile(pts, "spherical")
    r_pbc, z_pbc = project_to_profile(pts, "spherical", box_size=box)
    np.testing.assert_allclose(r_pbc, r_legacy, atol=1e-4)
    np.testing.assert_allclose(z_pbc, z_legacy)


def test_project_with_box_size_handles_droplet_straddling_boundary():
    """Atoms wrapped across the x=0 boundary must collapse onto sensible
    radii under the PBC-aware path, whereas the legacy arithmetic mean
    sees a spurious cluster centered in the empty middle of the box."""
    # Two tight rings of four atoms straddling the x=0 / x=L boundary on a
    # 10 Å box. Physically one ring of radius 0.5 centered on x=0.
    box = (10.0, 10.0)
    pts = np.array(
        [
            [0.5, 5.0, 0.0],
            [9.5, 5.0, 0.0],
            [0.0, 5.5, 0.0],
            [0.0, 4.5, 0.0],
        ]
    )
    r_pbc, _ = project_to_profile(pts, "spherical", box_size=box)
    # All atoms are at true radius 0.5 from the (wrapped) center.
    np.testing.assert_allclose(np.sort(r_pbc), np.full(4, 0.5), atol=1e-9)
    # Sanity check that this would have failed without the box-aware path:
    # the legacy mean lands near x=2.5, putting two atoms at r >= 7.
    r_legacy, _ = project_to_profile(pts, "spherical")
    assert float(np.max(r_legacy)) > 5.0


def test_project_cylinder_with_box_size_does_not_recenter_axial_axis():
    """The axial axis of a cylinder (x for cylinder_x, y for cylinder_y)
    must not be folded; only the cross-section axis is recentered, so the
    radial values match the legacy path on a mid-box cluster."""
    box = (20.0, 20.0)
    pts = _localized_cluster(box[0])
    for geom in ("cylinder_x", "cylinder_y"):
        r_legacy, _ = project_to_profile(pts, geom)
        r_pbc, _ = project_to_profile(pts, geom, box_size=box)
        np.testing.assert_allclose(r_pbc, r_legacy, atol=1e-4)
