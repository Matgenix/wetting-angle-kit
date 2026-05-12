import os
from typing import Any

from wetting_angle_kit.parser.parser_ase import AseWaterMoleculeFinder
from wetting_angle_kit.parser.parser_dump import DumpWaterMoleculeFinder
from wetting_angle_kit.parser.parser_xyz import XYZWaterMoleculeFinder


def get_water_parser(
    filename: str,
    particle_type_wall: Any,
    oxygen_type: Any,
    hydrogen_type: Any,
) -> Any:
    """Factory to select the correct water oxygen parser based on file extension."""
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".lammpstrj":
        return DumpWaterMoleculeFinder(
            filename, particle_type_wall, oxygen_type, hydrogen_type
        )
    elif ext in (".traj", ".ase"):
        return AseWaterMoleculeFinder(
            filename,
            particle_type_wall,
            oxygen_type=oxygen_type,
            hydrogen_type=hydrogen_type,
        )
    elif ext == ".xyz":
        return XYZWaterMoleculeFinder(
            filename,
            particle_type_wall,
            oxygen_type=oxygen_type,
            hydrogen_type=hydrogen_type,
        )
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: .lammpstrj, .traj/.ase, .xyz"
        )
