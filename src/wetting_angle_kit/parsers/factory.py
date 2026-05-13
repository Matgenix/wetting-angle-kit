import os
from typing import Any

from wetting_angle_kit.parsers.ase import AseWaterFinder
from wetting_angle_kit.parsers.lammps_dump import LammpsDumpWaterFinder
from wetting_angle_kit.parsers.xyz import XYZWaterFinder


def get_water_finder(
    filename: str,
    particle_type_wall: Any,
    oxygen_type: Any,
    hydrogen_type: Any,
) -> Any:
    """Factory to select the correct water oxygen finder based on file extension."""
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".lammpstrj":
        return LammpsDumpWaterFinder(
            filename, particle_type_wall, oxygen_type, hydrogen_type
        )
    elif ext in (".traj", ".ase"):
        return AseWaterFinder(
            filename,
            particle_type_wall,
            oxygen_type=oxygen_type,
            hydrogen_type=hydrogen_type,
        )
    elif ext == ".xyz":
        return XYZWaterFinder(
            filename,
            particle_type_wall,
            oxygen_type=oxygen_type,
            hydrogen_type=hydrogen_type,
        )
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: .lammpstrj, .traj/.ase, .xyz"
        )
