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
    """Load a LAMMPS dump file via OVITO and return the pipeline.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.

    Returns
    -------
    Any
        OVITO pipeline object (typed as Any because OVITO ships without stubs).
    """
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
    """Save a numpy array to a whitespace-delimited text file.

    Parameters
    ----------
    array : ndarray
        Array to save.
    filename : str
        Output file path.
    """
    np.savetxt(filename, array, fmt="%f")


def geometric_center(list_xyz_point: np.ndarray) -> np.ndarray:
    """Return the geometric center (mean position) of a point cloud.

    Parameters
    ----------
    list_xyz_point : ndarray, shape (N, 3)
        Cartesian coordinates of the points.

    Returns
    -------
    ndarray, shape (3,)
        Mean position vector.
    """
    return np.mean(list_xyz_point, axis=0)


def detect_parser_type(filename: str) -> str:
    """Infer the parser type from a trajectory file extension.

    Parameters
    ----------
    filename : str
        Path to the trajectory file.

    Returns
    -------
    str
        One of ``"dump"``, ``"ase"``, or ``"xyz"``.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".lammpstrj":
        return "dump"
    if ext in (".traj", ".ase"):
        return "ase"
    if ext == ".xyz":
        return "xyz"
    raise ValueError(f"Unsupported trajectory file format: {ext}")
