import os

import numpy as np
import pytest
from wetting_angle_kit.parser.parser_xyz import XYZParser

# Path to the test trajectory file
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__), "../trajectories/slice_10_mace_mlips_cylindrical_2_5.xyz"
)


# --- Fixture for XYZParser ---
@pytest.fixture
def xyz_parser():
    return XYZParser(TRAJECTORY_PATH)


# --- Test load_xyz_file ---
def test_load_xyz_file(xyz_parser):
    frames = xyz_parser.frames
    assert len(frames) > 0  # At least one frame should be loaded
    for frame in frames:
        assert "symbols" in frame
        assert "positions" in frame
        assert "lattice_matrix" in frame
        assert isinstance(frame["symbols"], np.ndarray)
        assert isinstance(frame["positions"], np.ndarray)
        assert frame["lattice_matrix"].shape == (3, 3)


# --- Test parse ---
def test_parse(xyz_parser):
    frame_index = 0
    positions = xyz_parser.parse(frame_index)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = [0, 1, 2]
    positions_subset = xyz_parser.parse(frame_index, indices)
    assert positions_subset.shape[0] == len(indices)


# --- Test parse_liquid_particles ---
def test_parse_liquid_particles(xyz_parser):
    frame_index = 0
    particle_type_liquid = ["O", "H"]
    liquid_positions = xyz_parser.parse_liquid_particles(
        particle_type_liquid, frame_index
    )
    assert isinstance(liquid_positions, np.ndarray)
    assert liquid_positions.shape[1] == 3  # x, y, z coordinates


# --- Test get_profile_coordinates ---
def test_get_cylindrical_coordinates(xyz_parser):
    frame_indices = [0, 1]
    r_values, z_values, n_frames = xyz_parser.get_profile_coordinates(frame_indices)
    assert isinstance(r_values, np.ndarray)
    assert isinstance(z_values, np.ndarray)
    assert n_frames == len(frame_indices)
    assert r_values.shape == z_values.shape

    # Test with atom_indices
    atom_indices = [0, 1, 2]
    r_values, z_values, _ = xyz_parser.get_profile_coordinates(
        frame_indices, atom_indices=atom_indices
    )
    assert r_values.size > 0
    assert z_values.size > 0


# --- Test box_length_max ---
def test_box_length_max(xyz_parser):
    frame_index = 0
    max_length = xyz_parser.box_length_max(frame_index)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test box_size_x and box_size_y ---
def test_box_size_x(xyz_parser):
    frame_index = 0
    box_size_x = xyz_parser.box_size_x(frame_index)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(xyz_parser):
    frame_index = 0
    box_size_y = xyz_parser.box_size_y(frame_index)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test frame_count ---
def test_frame_count(xyz_parser):
    total_frames = xyz_parser.frame_count()
    assert isinstance(total_frames, int)
    assert total_frames > 0
