import os
from typing import Any

import numpy as np


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


def _circular_mean_1d(coords: np.ndarray, box_length: float) -> float:
    """Periodic ("circular") mean of 1D coordinates on a box of length L.

    Maps each coordinate to a phase angle on a circle of circumference L,
    averages on the circle, and maps back to [0, L). This gives a meaningful
    center for a localized cluster regardless of where it sits relative to
    the periodic boundary -- including the half-in/half-out case where the
    plain arithmetic mean is wrong.

    Degenerate for a distribution that fills the box uniformly (the resulting
    angle is dominated by noise); callers must only apply it to axes along
    which the cluster is localized.
    """
    theta = 2.0 * np.pi * coords / box_length
    mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    return float((mean_angle % (2.0 * np.pi)) * box_length / (2.0 * np.pi))


def _confined_lateral_axes(droplet_geometry: str) -> tuple[int, ...]:
    """Lateral axes on which the droplet is localized (and therefore needs
    PBC-aware recentering). The axial direction of a cylinder is excluded
    because the atomic distribution along it fills the box."""
    if droplet_geometry == "spherical":
        return (0, 1)
    if droplet_geometry == "cylinder_y":
        return (0,)
    if droplet_geometry == "cylinder_x":
        return (1,)
    raise ValueError(
        f"Unknown droplet_geometry {droplet_geometry!r}; expected one of "
        "'spherical', 'cylinder_x', 'cylinder_y'."
    )


def recenter_droplet_pbc(
    positions: np.ndarray,
    droplet_geometry: str,
    box_size: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fold atomic positions into the minimum-image frame around the
    droplet's circular-mean center of mass.

    Use this on raw trajectory frames whose dynamics did not recenter the
    droplet (or recentered it but left atoms wrapped at the periodic
    boundary). After this call, atoms that belong to the same droplet form a
    single contiguous cluster around the returned center -- correct even
    when the droplet straddles a box face. The operation is idempotent on
    already-centered, unwrapped trajectories.

    Recentering is applied only on the *confined* lateral axes
    (cross-section of a cylinder, both x and y for a sphere). The axial
    direction of a cylinder is left untouched: the atomic distribution
    there fills the box, the circular mean is degenerate, and downstream
    analyses do not use a center along that axis.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian atomic positions for a single frame.
    droplet_geometry : str
        One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.
    box_size : (Lx, Ly)
        Lateral box lengths.

    Returns
    -------
    positions_folded : ndarray, shape (N, 3)
        Positions shifted by integer multiples of L on each confined axis so
        that the droplet forms a single cluster around ``com``. The axial
        axis of a cylinder and the z axis are returned unchanged.
    com : ndarray, shape (3,)
        Droplet center: circular-mean on confined axes, arithmetic mean on
        the others (axial / z). Lies in the lab frame; on confined axes it
        is mapped into ``[0, L)``.

    Notes
    -----
    The caller is responsible for ensuring the simulation box is large
    enough that the droplet does not overlap with its periodic image; this
    function does not validate box sizing.
    """
    if positions.size == 0:
        return positions.copy(), np.full(3, np.nan)

    folded = positions.copy()
    com = np.mean(positions, axis=0)  # default for axes we don't touch
    for axis in _confined_lateral_axes(droplet_geometry):
        L = float(box_size[axis])
        cm = _circular_mean_1d(positions[:, axis], L)
        d = positions[:, axis] - cm
        d -= L * np.round(d / L)  # minimum image around the circular COM
        folded[:, axis] = cm + d
        com[axis] = cm
    return folded, com
