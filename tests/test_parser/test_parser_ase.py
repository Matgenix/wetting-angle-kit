import os

import numpy as np
import pytest
from wetting_angle_kit.parsers.ase import AseParser

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




# --- Test box_size_x and box_size_y ---
def test_box_size_x(ase_parser):
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


# --- Test non-orthogonal cell rejection ---
def test_ase_parser_rejects_triclinic_cell(tmp_path):
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    atoms.set_cell([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 20.0]])
    path = tmp_path / "triclinic.traj"
    write(str(path), atoms)
    with pytest.raises(ValueError, match="Non-orthogonal"):
        AseParser(str(path))
    del ase


def test_ase_parser_box_sizes_match_lattice_norms(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    atoms.set_cell([[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 20.0]])
    path = tmp_path / "ortho.traj"
    write(str(path), atoms)
    parser = AseParser(str(path))
    assert parser.box_size_x(0) == pytest.approx(10.0)
    assert parser.box_size_y(0) == pytest.approx(12.0)
    assert parser.box_length_max(0) == pytest.approx(20.0)

