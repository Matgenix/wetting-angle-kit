import os
from unittest.mock import patch

import numpy as np
import pytest

from wetting_angle_kit.parsers.lammps_dump import LammpsDumpParser

# Path to the test trajectory file (LAMMPS dump format)
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__), "../trajectories/traj_spherical_drop_4k.lammpstrj"
)


# --- Fixture for LammpsDumpParser ---
@pytest.fixture
def dump_parser():
    return LammpsDumpParser(TRAJECTORY_PATH)


# --- Test ImportError ---
@patch(
    "ovito.io.import_file",
    side_effect=ImportError(
        "The 'ovito' package is required for LammpsDumpParser. Install with: "
        "pip install wetting_angle_kit[ovito]"
    ),
)
def test_dump_parser_no_ovito(mock_import_file):
    with pytest.raises(ImportError) as excinfo:
        LammpsDumpParser(TRAJECTORY_PATH)
    assert "The 'ovito' package is required for LammpsDumpParser" in str(excinfo.value)


# --- Test parse ---
def test_parse(dump_parser):
    frame_index = 0
    positions = dump_parser.parse(frame_index)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = np.array([1, 2])
    positions_subset = dump_parser.parse(frame_index, indices)
    assert positions_subset.shape[0] <= positions.shape[0]


# --- Test box_size_x and box_size_y ---
def test_box_size_x(dump_parser):
    frame_index = 0
    box_size_x = dump_parser.box_size_x(frame_index)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(dump_parser):
    frame_index = 0
    box_size_y = dump_parser.box_size_y(frame_index)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test box_length_max ---
def test_box_length_max(dump_parser):
    frame_index = 0
    max_length = dump_parser.box_length_max(frame_index)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test frame_count ---
def test_frame_count(dump_parser):
    total_frames = dump_parser.frame_count()
    assert isinstance(total_frames, int)
    assert total_frames > 0


# --- frame_tot is a deprecated alias for frame_count ---
def test_frame_tot_emits_deprecation_warning(dump_parser):
    with pytest.warns(DeprecationWarning, match="frame_tot is deprecated"):
        total = dump_parser.frame_tot()
    assert total == dump_parser.frame_count()


# --- Test non-orthogonal cell rejection ---
def _write_triclinic_dump(path):
    path.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        "0.0 10.0 5.0\n"
        "0.0 8.66 0.0\n"
        "0.0 20.0 0.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 1.0 1.0 1.0\n"
    )


def test_dump_parser_rejects_triclinic_cell(tmp_path):
    path = tmp_path / "triclinic.lammpstrj"
    _write_triclinic_dump(path)
    parser = LammpsDumpParser(str(path))
    # Validation is per-frame and lazy.
    with pytest.raises(ValueError, match="Non-orthogonal"):
        parser.parse(0)
