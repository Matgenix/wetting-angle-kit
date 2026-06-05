import pytest

from tests.conftest import trajectory_path
from wetting_angle_kit.parsers.ase import AseWaterFinder
from wetting_angle_kit.parsers.factory import get_water_finder
from wetting_angle_kit.parsers.lammps_dump import LammpsDumpWaterFinder
from wetting_angle_kit.parsers.xyz import XYZWaterFinder


def test_get_water_finder_lammpstrj():
    pytest.importorskip("ovito")
    finder = get_water_finder(
        trajectory_path("traj_spherical_drop_4k.lammpstrj"),
        oxygen_type=3,
        hydrogen_type=2,
    )
    assert isinstance(finder, LammpsDumpWaterFinder)


def test_get_water_finder_ase_traj():
    finder = get_water_finder(
        trajectory_path("slice_10_mace_mlips_cylindrical_2_5.traj"),
        oxygen_type="O",
        hydrogen_type="H",
    )
    assert isinstance(finder, AseWaterFinder)


def test_get_water_finder_xyz():
    finder = get_water_finder(
        trajectory_path("slice_10_mace_mlips_cylindrical_2_5.xyz"),
        oxygen_type="O",
        hydrogen_type="H",
    )
    assert isinstance(finder, XYZWaterFinder)


def test_get_water_finder_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        get_water_finder(
            "/tmp/does_not_matter.pdb",
            oxygen_type="O",
            hydrogen_type="H",
        )
