import os
from typing import Any

import numpy as np

#: Droplet geometry strings accepted across analyzers and parsers.
VALID_DROPLET_GEOMETRIES = ("spherical", "cylinder_x", "cylinder_y")


def validate_droplet_geometry(droplet_geometry: str) -> None:
    """Raise ``ValueError`` if ``droplet_geometry`` is not one of the
    supported values: ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.
    """
    if droplet_geometry not in VALID_DROPLET_GEOMETRIES:
        raise ValueError(
            f"Unknown droplet_geometry {droplet_geometry!r}. "
            f"Expected one of {VALID_DROPLET_GEOMETRIES}."
        )


def load_dump_ovito(filepath: str) -> Any:
    try:
        from ovito.io import import_file
    except ImportError as e:  # add exception chaining
        raise ImportError(
            "The 'ovito' package is required for load dump_ovito. Install it with: "
            "pip install wetting_angle_kit[ovito]"
        ) from e
    pipeline = import_file(filepath)
    # Add necessary modifiers
    return pipeline


def save_array_as_txt(array: np.ndarray, filename: str) -> None:
    np.savetxt(filename, array, fmt="%f")


def geometric_center(list_xyz_point: np.ndarray) -> np.ndarray:
    return np.mean(list_xyz_point, axis=0)


def detect_parser_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".lammpstrj":
        return "dump"
    if ext in (".traj", ".ase"):
        return "ase"
    if ext == ".xyz":
        return "xyz"
    raise ValueError(f"Unsupported trajectory file format: {ext}")
