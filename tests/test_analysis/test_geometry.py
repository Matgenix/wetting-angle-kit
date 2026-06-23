"""Unit tests for :class:`DropletGeometry`."""

import numpy as np
import pytest

from wetting_angle_kit.analysis.geometry import DropletGeometry

# --- Construction & validation ----------------------------------------------


def test_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="droplet_geometry must be one of"):
        DropletGeometry(name="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["spherical", "cylinder_x", "cylinder_y"])
def test_accepts_valid_names(name: str) -> None:
    geom = DropletGeometry(name=name)  # type: ignore[arg-type]
    assert geom.name == name


# --- coerce() ---------------------------------------------------------------


def test_coerce_returns_instance_unchanged() -> None:
    g = DropletGeometry.coerce("spherical")
    assert DropletGeometry.coerce(g) is g


def test_coerce_from_string() -> None:
    g = DropletGeometry.coerce("cylinder_y")
    assert isinstance(g, DropletGeometry)
    assert g.name == "cylinder_y"


# --- Predicates -------------------------------------------------------------


def test_is_spherical_and_is_cylinder() -> None:
    sph = DropletGeometry.coerce("spherical")
    cyx = DropletGeometry.coerce("cylinder_x")
    cyy = DropletGeometry.coerce("cylinder_y")
    assert sph.is_spherical and not sph.is_cylinder
    assert cyx.is_cylinder and not cyx.is_spherical
    assert cyy.is_cylinder and not cyy.is_spherical


# --- cylinder_axis ----------------------------------------------------------


def test_cylinder_axis() -> None:
    assert DropletGeometry.coerce("cylinder_x").cylinder_axis == "x"
    assert DropletGeometry.coerce("cylinder_y").cylinder_axis == "y"
    assert DropletGeometry.coerce("spherical").cylinder_axis is None


# --- Coordinate-frame swaps -------------------------------------------------


def test_to_internal_coords_swaps_for_cylinder_x() -> None:
    """``cylinder_x`` swaps x↔y so the cylinder axis ends up on y."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = DropletGeometry.coerce("cylinder_x").to_internal_coords(coords)
    np.testing.assert_array_equal(out, [[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]])


def test_to_internal_coords_is_identity_for_spherical_and_cylinder_y() -> None:
    coords = np.array([[1.0, 2.0, 3.0]])
    for name in ("spherical", "cylinder_y"):
        out = DropletGeometry.coerce(name).to_internal_coords(coords)
        np.testing.assert_array_equal(out, coords)


def test_to_user_coords_is_inverse_of_to_internal() -> None:
    """The swap is an involution: roundtrip restores the input."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for name in ("spherical", "cylinder_x", "cylinder_y"):
        g = DropletGeometry.coerce(name)
        np.testing.assert_array_equal(
            g.to_user_coords(g.to_internal_coords(coords)), coords
        )
