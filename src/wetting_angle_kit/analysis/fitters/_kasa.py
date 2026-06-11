"""Algebraic (Kasa) circle/sphere fit helpers + cap-angle utilities.

Shared by both :class:`_SlicingFitter` (2D circle on per-slice points)
and :class:`_WholeFitter` (2D circle for cylinder droplets, 3D sphere
for spherical droplets).
"""

import numpy as np


def _kasa_circle_fit_2d(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit in 2D.

    Linearises ``(x - xc)^2 + (z - zc)^2 = R^2`` into
    ``2 xc x + 2 zc z + c = x^2 + z^2`` with ``c = R^2 - xc^2 - zc^2``
    and solves with :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    x, z : ndarray
        2D point coordinates.

    Returns
    -------
    (xc, zc, R) : tuple of float
        Fitted circle centre and radius.

    Raises
    ------
    np.linalg.LinAlgError
        If the points are collinear (rank-deficient system).
    ValueError
        If the algebraic solution gives a non-positive ``R^2``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    a_matrix = np.column_stack((2.0 * x, 2.0 * z, np.ones_like(x)))
    rhs = x * x + z * z
    sol, _, _, _ = np.linalg.lstsq(a_matrix, rhs, rcond=None)
    xc, zc, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + xc * xc + zc * zc
    if r_sq <= 0.0:
        raise ValueError(
            f"Algebraic circle fit produced non-positive R^2 ({r_sq:.3g}); "
            "the points are likely degenerate."
        )
    return xc, zc, float(np.sqrt(r_sq))


def _kasa_sphere_fit_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) least-squares sphere fit in 3D.

    Linearises ``(x - xc)^2 + (y - yc)^2 + (z - zc)^2 = R^2`` into
    ``2 xc x + 2 yc y + 2 zc z + c = x^2 + y^2 + z^2`` with
    ``c = R^2 - xc^2 - yc^2 - zc^2`` and solves with
    :func:`numpy.linalg.lstsq`.

    Returns ``(xc, yc, zc, R)``.

    Raises
    ------
    np.linalg.LinAlgError
        If the input points are co-planar (rank-deficient system).
    ValueError
        If the algebraic solution gives a non-positive ``R^2``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    a_matrix = np.column_stack((2.0 * x, 2.0 * y, 2.0 * z, np.ones_like(x)))
    rhs = x * x + y * y + z * z
    sol, _, _, _ = np.linalg.lstsq(a_matrix, rhs, rcond=None)
    xc, yc, zc, c = (float(sol[0]), float(sol[1]), float(sol[2]), float(sol[3]))
    r_sq = c + xc * xc + yc * yc + zc * zc
    if r_sq <= 0.0:
        raise ValueError(
            f"Algebraic sphere fit produced non-positive R^2 ({r_sq:.3g}); "
            "the points are likely degenerate."
        )
    return xc, yc, zc, float(np.sqrt(r_sq))


def _whole_fit_one(
    points: np.ndarray, *, spherical: bool
) -> tuple[np.ndarray, float, float]:
    """Fit a sphere (spherical=True) or cylinder (spherical=False) to ``points``.

    Returns ``(popt_no_wall, R, zc)`` where ``popt_no_wall`` is
    ``[xc, yc, zc, R]`` for spherical or ``[xc, zc, R]`` for cylinder.
    The cylinder fit drops the ``y`` column and fits a 2D circle in
    ``(x, z)``.
    """
    if spherical:
        xc, yc, zc, R = _kasa_sphere_fit_3d(points[:, 0], points[:, 1], points[:, 2])
        return np.array([xc, yc, zc, R]), R, zc
    xc, zc, R = _kasa_circle_fit_2d(points[:, 0], points[:, 2])
    return np.array([xc, zc, R]), R, zc


def _angle_from_cap(z_wall: float, zc: float, R: float) -> float | None:
    """Contact angle from ``cos θ = (z_wall - zc) / R`` or None if no intersection."""
    delta_z = z_wall - zc
    if abs(delta_z) >= R:
        return None
    return float(np.degrees(np.arccos(delta_z / R)))
