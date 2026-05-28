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
    """Return the appropriate water oxygen finder for a given trajectory file.

    Parameters
    ----------
    filename : str
        Path to trajectory file; extension determines the finder class.
    particle_type_wall : Any
        Wall particle type identifiers forwarded to the finder constructor.
    oxygen_type : Any
        Oxygen type identifier (symbol or integer depending on file format).
    hydrogen_type : Any
        Hydrogen type identifier (symbol or integer depending on file format).

    Returns
    -------
    LammpsDumpWaterFinder | AseWaterFinder | XYZWaterFinder
        Finder instance matching the file format.
    """
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
