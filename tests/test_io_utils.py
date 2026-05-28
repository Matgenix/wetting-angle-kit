"""Unit tests for :mod:`wetting_angle_kit.io_utils`."""

import os
import sys
from unittest import mock

import numpy as np
import pytest

from wetting_angle_kit.io_utils import (
    VALID_DROPLET_GEOMETRIES,
    assert_orthogonal_cell,
    detect_parser_type,
    geometric_center,
    recenter_droplet_pbc,
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


# --- assert_orthogonal_cell ---


def test_assert_orthogonal_cell_accepts_diagonal():
    cell = np.diag([10.0, 12.0, 30.0])
    assert_orthogonal_cell(cell)


def test_assert_orthogonal_cell_accepts_tiny_off_diagonal():
    cell = np.diag([10.0, 12.0, 30.0]) + 1e-9
    np.fill_diagonal(cell, [10.0, 12.0, 30.0])
    assert_orthogonal_cell(cell)


def test_assert_orthogonal_cell_rejects_triclinic():
    cell = np.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 20.0]])
    with pytest.raises(ValueError, match="Non-orthogonal"):
        assert_orthogonal_cell(cell)


def test_assert_orthogonal_cell_rejects_rotated_cube():
    # Cubic but rotated 45° in xy: lattice vectors not aligned with axes.
    s = 10.0 / np.sqrt(2)
    cell = np.array([[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 20.0]])
    with pytest.raises(ValueError, match="Non-orthogonal"):
        assert_orthogonal_cell(cell)


def test_assert_orthogonal_cell_context_appears_in_message():
    cell = np.array([[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    with pytest.raises(ValueError, match=r"Frame 7:"):
        assert_orthogonal_cell(cell, context="Frame 7")


def test_assert_orthogonal_cell_bad_shape():
    with pytest.raises(ValueError, match="3x3"):
        assert_orthogonal_cell(np.zeros((4, 4)))


# --- recenter_droplet_pbc ---

# A small box used by the recentering tests below.
_BOX = (10.0, 10.0)


def test_recenter_localized_cluster_matches_arithmetic_mean():
    """A cluster comfortably inside the box: COM ~ arithmetic mean and
    positions are unchanged (no wraps needed)."""
    positions = np.array([[4.0, 5.0, 0.0], [5.0, 5.0, 1.0], [6.0, 5.0, 2.0]])
    folded, com = recenter_droplet_pbc(positions, "spherical", box_size=_BOX)
    np.testing.assert_allclose(folded, positions)
    np.testing.assert_allclose(com[:2], np.mean(positions, axis=0)[:2], atol=1e-9)
    # z is the arithmetic mean by definition (untouched axis).
    assert com[2] == pytest.approx(1.0)


def test_recenter_cluster_straddling_x_boundary():
    """Atoms split across the x=0 / x=L boundary: positions are unwrapped
    into a single contiguous cluster, and the centered extent stays small."""
    # Two atoms at x=0.5 and two at x=9.5 (L=10): physically one tight cluster
    # of radius 0.5 centered on x=0 (=x=10).
    positions = np.array(
        [
            [0.5, 5.0, 0.0],
            [0.5, 5.0, 1.0],
            [9.5, 5.0, 2.0],
            [9.5, 5.0, 3.0],
        ]
    )
    folded, com = recenter_droplet_pbc(positions, "spherical", box_size=_BOX)
    # The folded x-coordinates must form a tight cluster (max - min <= 1.0),
    # which is impossible without wrapping atoms across the boundary.
    assert float(np.ptp(folded[:, 0])) <= 1.0 + 1e-9
    # The naive mean would put the COM near x=5 (the empty middle). The
    # circular mean should put it near the boundary (0 or 10 mod 10).
    cm_x_mod = float(com[0] % _BOX[0])
    assert min(cm_x_mod, _BOX[0] - cm_x_mod) <= 0.5 + 1e-9


def test_recenter_is_idempotent_on_already_centered_cluster():
    """Applying recentering twice gives (modulo translation by L) the same
    folded positions as applying it once."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(low=4.0, high=6.0, size=(20, 3))
    folded_once, _ = recenter_droplet_pbc(positions, "spherical", box_size=_BOX)
    folded_twice, _ = recenter_droplet_pbc(folded_once, "spherical", box_size=_BOX)
    # Allow a uniform integer-L shift between the two folds (the circular
    # mean is defined modulo L, so the absolute origin may differ).
    diff = folded_twice - folded_once
    for axis in (0, 1):
        shifts = diff[:, axis] / _BOX[axis]
        np.testing.assert_allclose(shifts, np.round(shifts), atol=1e-9)


@pytest.mark.parametrize(
    "geometry, untouched_axis",
    [("cylinder_x", 0), ("cylinder_y", 1)],
)
def test_recenter_leaves_cylinder_axial_axis_untouched(geometry, untouched_axis):
    """The axial direction of a cylinder must not be folded -- its
    coordinates are returned unchanged and the COM uses the arithmetic mean
    on that axis."""
    # Atoms spread along the axial axis and tightly clustered on the other.
    rng = np.random.default_rng(1)
    positions = np.column_stack(
        [
            rng.uniform(0.0, _BOX[0], size=20),  # x
            np.full(20, 5.0) + rng.normal(scale=0.1, size=20),  # y near 5
            rng.uniform(0.0, 5.0, size=20),  # z
        ]
    )
    if untouched_axis == 1:
        # cylinder_y: y is axial, x is confined -> swap which is spread.
        positions[:, [0, 1]] = positions[:, [1, 0]]
    folded, com = recenter_droplet_pbc(positions, geometry, box_size=_BOX)
    np.testing.assert_allclose(folded[:, untouched_axis], positions[:, untouched_axis])
    assert com[untouched_axis] == pytest.approx(
        float(np.mean(positions[:, untouched_axis]))
    )


def test_recenter_does_not_validate_box_sizing():
    """recenter_droplet_pbc trusts the caller on box sizing: it never raises
    on geometric grounds. Even a cluster that fills most of the box returns
    a valid (if physically borderline) result with all folded coordinates
    bounded by L/2 of the chosen center."""
    rng = np.random.default_rng(2)
    positions = np.column_stack(
        [
            rng.uniform(0.0, _BOX[0], size=200),
            rng.uniform(0.0, _BOX[1], size=200),
            np.zeros(200),
        ]
    )
    folded, com = recenter_droplet_pbc(positions, "spherical", box_size=_BOX)
    for axis in (0, 1):
        d = folded[:, axis] - com[axis]
        assert float(np.max(np.abs(d))) <= 0.5 * _BOX[axis] + 1e-9


def test_recenter_empty_positions():
    """Empty input returns an empty array and a NaN COM (no atoms to mean)."""
    folded, com = recenter_droplet_pbc(np.zeros((0, 3)), "spherical", box_size=_BOX)
    assert folded.shape == (0, 3)
    assert np.all(np.isnan(com))
