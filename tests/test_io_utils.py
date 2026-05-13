"""Unit tests for :mod:`wetting_angle_kit.io_utils`."""

from __future__ import annotations

import os
import sys
from unittest import mock

import numpy as np
import pytest
from wetting_angle_kit.io_utils import (
    VALID_DROPLET_GEOMETRIES,
    detect_parser_type,
    geometric_center,
    save_array_as_txt,
    validate_droplet_geometry,
)

# --- detect_parser_type ---


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("traj.lammpstrj", "dump"),
        ("TRAJ.LAMMPSTRJ", "dump"),  # case-insensitive
        ("foo.traj", "ase"),
        ("foo.ase", "ase"),
        ("structure.xyz", "xyz"),
        ("/path/with.dots/trajectory.lammpstrj", "dump"),
    ],
)
def test_detect_parser_type_supported(filename, expected):
    assert detect_parser_type(filename) == expected


@pytest.mark.parametrize("filename", ["foo.dump", "foo.txt", "foo", "foo.tar.gz"])
def test_detect_parser_type_rejects_unknown(filename):
    with pytest.raises(ValueError, match="Unsupported trajectory file format"):
        detect_parser_type(filename)


# --- geometric_center ---


def test_geometric_center_simple():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    center = geometric_center(points)
    np.testing.assert_array_equal(center, np.array([1.0, 2.0, 3.0]))


def test_geometric_center_single_point():
    points = np.array([[1.5, -2.0, 3.7]])
    center = geometric_center(points)
    np.testing.assert_array_equal(center, np.array([1.5, -2.0, 3.7]))


# --- save_array_as_txt ---


def test_save_array_as_txt_roundtrip(tmp_path):
    target = tmp_path / "values.txt"
    data = np.array([[1.0, 2.0], [3.5, 4.25]])
    save_array_as_txt(data, str(target))
    assert target.exists()
    loaded = np.loadtxt(target)
    np.testing.assert_allclose(loaded, data)


# --- validate_droplet_geometry ---


@pytest.mark.parametrize("geom", VALID_DROPLET_GEOMETRIES)
def test_validate_droplet_geometry_accepts_valid(geom):
    # Should not raise.
    validate_droplet_geometry(geom)


@pytest.mark.parametrize("bad", ["spheric", "cylinder", "Cylinder_y", "", "sphere"])
def test_validate_droplet_geometry_rejects_invalid(bad):
    with pytest.raises(ValueError, match="Unknown droplet_geometry"):
        validate_droplet_geometry(bad)


# --- load_dump_ovito (only test the ImportError path; calling it for real
# requires ovito and a trajectory, which the other test modules cover) ---


def test_load_dump_ovito_raises_when_ovito_missing():
    from wetting_angle_kit import io_utils

    # Block ovito imports for the duration of this test.
    with mock.patch.dict(sys.modules, {"ovito": None, "ovito.io": None}):
        with pytest.raises(ImportError, match="ovito"):
            io_utils.load_dump_ovito("/nonexistent.lammpstrj")


def test_valid_droplet_geometries_constant_is_a_tuple():
    # Constant should be a frozen tuple-like sequence so callers cannot
    # mutate the package-level whitelist accidentally.
    assert isinstance(VALID_DROPLET_GEOMETRIES, tuple)
    assert set(VALID_DROPLET_GEOMETRIES) == {"spherical", "cylinder_x", "cylinder_y"}


# --- Round-trip with detect + temp file ---


def test_detect_parser_type_resolves_relative_path(tmp_path):
    # Path that contains directory traversal should still resolve by extension.
    f = tmp_path / "sub.dir" / "data.xyz"
    os.makedirs(f.parent, exist_ok=True)
    f.write_text("")
    assert detect_parser_type(str(f)) == "xyz"
