"""Unit tests for the shared (r, z) projection used by
:class:`BaseParser.get_profile_coordinates`.

These exercise all three droplet geometries (including ``cylinder_x``
which was not previously covered by any test).
"""

from __future__ import annotations

import numpy as np
import pytest
from wetting_angle_kit.parsers.base import project_to_profile


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
