import numpy as np
import pytest

from wetting_angle_kit.parsers.ase import AseWallParser, AseWaterFinder
from wetting_angle_kit.parsers.xyz import XYZWaterFinder


def _write_water_xyz(path, with_lattice=True):
    """Write a small XYZ trajectory with two water molecules and one wall C atom.

    Atoms (Å):
      O1 at (0, 0, 0), H1 at (0.95, 0, 0), H2 at (-0.95, 0, 0)   # water 1
      O2 at (5, 5, 5), H3 at (5.95, 5, 5), H4 at (4.05, 5, 5)    # water 2
      C  at (10, 10, 10)                                          # wall

    With these positions, only the two oxygens have exactly two hydrogens
    within ``oh_cutoff = 1.2``.
    """
    if with_lattice:
        header = (
            '7\nLattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" '
            "Properties=species:S:1:pos:R:3\n"
        )
    else:
        header = "7\nNo lattice here\n"
    lines = [
        "O 0.0 0.0 0.0",
        "H 0.95 0.0 0.0",
        "H -0.95 0.0 0.0",
        "O 5.0 5.0 5.0",
        "H 5.95 5.0 5.0",
        "H 4.05 5.0 5.0",
        "C 10.0 10.0 10.0",
    ]
    path.write_text(header + "\n".join(lines) + "\n")


@pytest.fixture
def water_xyz(tmp_path):
    p = tmp_path / "water.xyz"
    _write_water_xyz(p, with_lattice=True)
    return str(p)


def test_xyz_water_finder_identifies_oxygens(water_xyz):
    finder = XYZWaterFinder(water_xyz, oxygen_type="O", hydrogen_type="H")
    indices = finder.get_water_oxygen_indices(0)
    # Two oxygens are at indices 0 and 3 in the input order.
    assert sorted(indices.tolist()) == [0, 3]


def test_xyz_water_finder_positions(water_xyz):
    finder = XYZWaterFinder(water_xyz)
    positions = finder.get_water_oxygen_positions(0)
    assert positions.shape == (2, 3)
    # Centers of the two waters.
    assert {tuple(p) for p in positions.tolist()} == {(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)}


def test_xyz_water_finder_no_water_returns_empty(tmp_path):
    p = tmp_path / "no_water.xyz"
    # One isolated oxygen (no hydrogens at all)
    p.write_text(
        '1\nLattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
        "Properties=species:S:1:pos:R:3\nO 0.0 0.0 0.0\n"
    )
    finder = XYZWaterFinder(str(p))
    positions = finder.get_water_oxygen_positions(0)
    assert positions.shape == (0, 3)


def test_xyz_water_finder_parse_filters_liquid(water_xyz):
    finder = XYZWaterFinder(water_xyz)
    positions = finder.parse(["O", "H"], 0)
    # Six liquid atoms (2 O + 4 H), wall (C) excluded.
    assert positions.shape == (6, 3)


def test_xyz_water_finder_box_length_max(water_xyz):
    finder = XYZWaterFinder(water_xyz)
    assert finder.box_length_max(0) == pytest.approx(20.0)


def test_xyz_water_finder_box_length_max_without_lattice_raises(tmp_path):
    p = tmp_path / "no_lattice.xyz"
    _write_water_xyz(p, with_lattice=False)
    finder = XYZWaterFinder(str(p))
    with pytest.raises(ValueError, match="No Lattice="):
        finder.box_length_max(0)


def _write_water_extxyz(path):
    """Write a small extended-XYZ readable by ASE with two waters + a wall C."""
    path.write_text(
        "7\n"
        'Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "O 0.0 0.0 0.0\n"
        "H 0.95 0.0 0.0\n"
        "H -0.95 0.0 0.0\n"
        "O 5.0 5.0 5.0\n"
        "H 5.95 5.0 5.0\n"
        "H 4.05 5.0 5.0\n"
        "C 10.0 10.0 10.0\n"
    )


@pytest.fixture
def water_extxyz(tmp_path):
    p = tmp_path / "water.extxyz"
    _write_water_extxyz(p)
    return str(p)


def test_ase_water_finder_identifies_oxygens(water_extxyz):
    finder = AseWaterFinder(
        water_extxyz,
        oxygen_type="O",
        hydrogen_type="H",
    )
    indices = finder.get_water_oxygen_indices(0)
    assert sorted(indices.tolist()) == [0, 3]


def test_ase_water_finder_positions(water_extxyz):
    finder = AseWaterFinder(water_extxyz)
    positions = finder.get_water_oxygen_positions(0)
    assert positions.shape == (2, 3)


def test_ase_wall_parser_extracts_wall(water_extxyz):
    parser = AseWallParser(water_extxyz, liquid_particle_types=["O", "H"])
    wall = parser.parse(0)
    assert wall.shape == (1, 3)
    assert np.allclose(wall[0], [10.0, 10.0, 10.0])


def test_ase_wall_parser_with_indices(water_extxyz):
    parser = AseWallParser(water_extxyz, liquid_particle_types=["O", "H"])
    wall = parser.parse(0, indices=[0])
    assert wall.shape == (1, 3)


def test_ase_wall_parser_find_highest_wall_particle(water_extxyz):
    parser = AseWallParser(water_extxyz, liquid_particle_types=["O", "H"])
    assert parser.find_highest_wall_particle(0) == pytest.approx(10.0)


def test_ase_wall_parser_box_sizes_and_frame_count(water_extxyz):
    parser = AseWallParser(water_extxyz, liquid_particle_types=["O", "H"])
    assert parser.box_size_x(0) == pytest.approx(20.0)
    assert parser.box_size_y(0) == pytest.approx(20.0)
    assert parser.box_length_max(0) == pytest.approx(20.0)
    assert parser.frame_count() == 1


def test_ase_wall_parser_find_highest_wall_part_emits_deprecation(water_extxyz):
    parser = AseWallParser(water_extxyz, liquid_particle_types=["O", "H"])
    with pytest.warns(DeprecationWarning, match="find_highest_wall_part is deprecated"):
        z = parser.find_highest_wall_part(0)
    assert z == pytest.approx(10.0)
