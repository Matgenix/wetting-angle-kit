import os

import numpy as np
import pytest

from wetting_angle_kit.parser.parser_ase import AseParser

# Path to the test trajectory file (ASE format)
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__),
    "../trajectories/slice_10_mace_mlips_cylindrical_2_5.traj",
)


# --- Fixture for AseParser ---
@pytest.fixture
def ase_parser():
    return AseParser(TRAJECTORY_PATH)


# --- Test parse ---
def test_parse(ase_parser):
    frame_index = 0
    positions = ase_parser.parse(frame_index)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = [0, 1, 2]
    positions_subset = ase_parser.parse(frame_index, indices)
    assert positions_subset.shape[0] == len(indices)


# --- Test parse_liquid_particles ---
def test_parse_liquid_particles(ase_parser):
    frame_index = 0
    particle_type_liquid = ["O", "H"]
    liquid_positions = ase_parser.parse_liquid_particles(
        particle_type_liquid, frame_index
    )
    assert isinstance(liquid_positions, np.ndarray)
    assert liquid_positions.shape[1] == 3  # x, y, z coordinates


# --- Test get_profile_coordinates ---
def test_get_cylindrical_coordinates(ase_parser, capsys):
    frame_indices = [0, 1]
    r_values, z_values, n_frames = ase_parser.get_profile_coordinates(frame_indices)
    assert isinstance(r_values, np.ndarray)
    assert isinstance(z_values, np.ndarray)
    assert n_frames == len(frame_indices)
    assert r_values.shape == z_values.shape

    # Test with atom_indices
    atom_indices = [0, 1, 2]
    r_values, z_values, _ = ase_parser.get_profile_coordinates(
        frame_indices, atom_indices=atom_indices
    )
    assert r_values.size > 0
    assert z_values.size > 0

    # Check print output
    captured = capsys.readouterr()
    assert "r range:" in captured.out
    assert "z range:" in captured.out


# --- Test box_size_x and box_size_y ---
def test_box_size_x(ase_parser, capsys):
    frame_index = 0
    box_size_x = ase_parser.box_size_x(frame_index)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(ase_parser):
    frame_index = 0
    box_size_y = ase_parser.box_size_y(frame_index)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test box_length_max ---
def test_box_length_max(ase_parser):
    frame_index = 0
    max_length = ase_parser.box_length_max(frame_index)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test frame_count ---
def test_frame_count(ase_parser):
    total_frames = ase_parser.frame_count()
    assert isinstance(total_frames, int)
    assert total_frames > 0


# --- Test droplet_geometry in get_profile_coordinates ---
def test_get_cylindrical_coordinates_type_model(ase_parser):
    frame_indices = [0]
    r_values, z_values, _ = ase_parser.get_profile_coordinates(
        frame_indices, droplet_geometry="spherical"
    )
    assert isinstance(r_values, np.ndarray)
    assert isinstance(z_values, np.ndarray)
