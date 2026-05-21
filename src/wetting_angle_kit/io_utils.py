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


def ovito_cell_vectors(data: Any) -> np.ndarray:
    """Return the 3x3 lattice matrix (lattice vectors as columns) from an
    OVITO ``DataCollection``. OVITO's ``cell.matrix`` is 3x4: the first
    three columns are the lattice vectors and the fourth is the origin.
    """
    return np.asarray(data.cell.matrix)[:, :3]


def assert_orthogonal_cell(
    cell: np.ndarray, *, tol: float = 1e-6, context: str = ""
) -> None:
    """Raise ``ValueError`` if a 3x3 cell matrix is not axis-aligned orthogonal.

    The check is convention-independent: an orthogonal cell whose lattice
    vectors are aligned with the x, y, z axes is diagonal whether the
    vectors are stored as rows (ASE/extxyz) or columns (OVITO).

    Parameters
    ----------
    cell : ndarray, shape (3, 3)
        Cell matrix containing the three lattice vectors.
    tol : float, default 1e-6
        Relative tolerance: off-diagonal entries are accepted if their
        magnitude is below ``tol * max(|cell|)``.
    context : str, optional
        Description prepended to the error message (e.g. ``"Frame 3"``).
    """
    cell_arr = np.asarray(cell, dtype=float)
    if cell_arr.shape != (3, 3):
        raise ValueError(f"Cell matrix must be 3x3, got shape {cell_arr.shape}.")
    off_diag = cell_arr - np.diag(np.diag(cell_arr))
    scale = max(1.0, float(np.max(np.abs(cell_arr))))
    if float(np.max(np.abs(off_diag))) > tol * scale:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Non-orthogonal simulation cells are not supported. "
            "Provide a trajectory whose lattice vectors are aligned with the "
            "x, y, z axes."
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
        OVITO pipeline object (typed as Any because OVITO lacks Python type stubs).
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


def project_to_profile(
    positions: np.ndarray, droplet_geometry: str
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D atomic positions onto the (r, z) plane used by analyzers.

    The lateral coordinates are centered on their per-frame center of mass
    before projection; the vertical (z) coordinate is left in lab frame.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian atomic positions for a single frame.
    droplet_geometry : str
        One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.

    Returns
    -------
    r_values : ndarray, shape (N,)
        Radial coordinate: |x_centered| for cylinder_y, |y_centered| for
        cylinder_x, sqrt(x_centered**2 + y_centered**2) for spherical.
    z_values : ndarray, shape (N,)
        Vertical coordinate (lab-frame z, not centered).
    """
    validate_droplet_geometry(droplet_geometry)
    if positions.size == 0:
        return np.empty(0), np.empty(0)
    x_cm = np.mean(positions, axis=0)
    x_centered = positions - x_cm
    # z stays in lab frame; analyzers need absolute heights to locate the wall.
    z_values = positions[:, 2]
    if droplet_geometry == "cylinder_y":
        r_values = np.abs(x_centered[:, 0])
    elif droplet_geometry == "cylinder_x":
        r_values = np.abs(x_centered[:, 1])
    else:  # droplet_geometry == "spherical"
        r_values = np.sqrt(x_centered[:, 0] ** 2 + x_centered[:, 1] ** 2)
    return r_values, z_values
