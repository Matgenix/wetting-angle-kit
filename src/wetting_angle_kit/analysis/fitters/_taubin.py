"""Algebraic (Taubin) circle/sphere fit helpers + cap-angle utilities.

Shared by both :class:`_SlicingFitter` (2D circle on per-slice points)
and :class:`_WholeFitter` (2D circle for cylinder droplets, 3D sphere
for spherical droplets).

A droplet cap is only ever a *partial* arc — the liquid-vapor surface,
never the full circle/sphere — and on a short, noisy arc the recovered
radius feeds straight through ``cos θ = (z_wall - z_c) / R`` into the
contact angle. The Taubin fit normalises the algebraic residual by its
gradient, which keeps the recovered radius near-unbiased on partial arcs
(matching a full geometric orthogonal-distance fit) while staying
closed-form: no initial guess, no iteration. ``benchmarks/bench_circle_fit.py``
quantifies the radius/angle accuracy and per-fit timing.
"""

import numpy as np


def _taubin_fit(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Taubin algebraic fit of a circle (2D) or sphere (3D) to ``coords``.

    Uses the SVD form of Taubin's method (Chernov): the data are centred,
    the squared-radius column is mean-subtracted and scaled, and the
    smallest right singular vector of the resulting design matrix gives
    the algebraic coefficients. This is numerically stable, dimension
    general, and guarantees a positive ``R^2``.

    Parameters
    ----------
    coords : ndarray, shape (N, D)
        Point coordinates, with ``D == 2`` (circle) or ``D == 3``
        (sphere).

    Returns
    -------
    (center, R) : tuple of (ndarray, float)
        Fitted centre (length ``D``) and radius.

    Raises
    ------
    np.linalg.LinAlgError
        If the SVD fails to converge.
    ValueError
        If there are too few points, all points coincide, or the points
        are collinear/coplanar (degenerate, near-zero curvature).
    """
    coords = np.asarray(coords, dtype=float)
    n, dim = coords.shape
    if n < dim + 1:
        raise ValueError(
            f"Taubin fit needs at least {dim + 1} points for a {dim}D fit; got {n}."
        )
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    sq = np.einsum("ij,ij->i", centered, centered)
    sq_mean = float(sq.mean())
    if sq_mean <= 0.0:
        raise ValueError("Taubin fit: all points coincide.")
    # Mean-subtracted, scaled squared-radius column + the centred
    # coordinates; the smallest right singular vector minimises the
    # Taubin (gradient-normalised) algebraic distance.
    z0 = (sq - sq_mean) / (2.0 * np.sqrt(sq_mean))
    design = np.column_stack([z0, centered])
    _, _, vt = np.linalg.svd(design, full_matrices=False)
    v = vt[-1]
    a0 = v[0] / (2.0 * np.sqrt(sq_mean))
    if abs(a0) < 1e-12:
        raise ValueError(
            "Taubin fit: near-zero curvature; the points are likely "
            "collinear (2D) or coplanar (3D)."
        )
    center_centered = -v[1:] / (2.0 * a0)
    # R^2 = |center|^2 + mean(|r|^2) in the centred frame; always > 0.
    r_sq = float(center_centered @ center_centered) + sq_mean
    center = center_centered + centroid
    return center, float(np.sqrt(r_sq))


def _taubin_circle_fit_2d(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Taubin algebraic least-squares circle fit in 2D.

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
        If the SVD fails to converge.
    ValueError
        If the points are degenerate (too few, coincident, or collinear).
    """
    center, r = _taubin_fit(np.column_stack((np.asarray(x), np.asarray(z))))
    return float(center[0]), float(center[1]), r


def _taubin_sphere_fit_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, float, float]:
    """Taubin algebraic least-squares sphere fit in 3D.

    Returns ``(xc, yc, zc, R)``.

    Raises
    ------
    np.linalg.LinAlgError
        If the SVD fails to converge.
    ValueError
        If the points are degenerate (too few, coincident, or coplanar).
    """
    center, r = _taubin_fit(
        np.column_stack((np.asarray(x), np.asarray(y), np.asarray(z)))
    )
    return float(center[0]), float(center[1]), float(center[2]), r


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
        xc, yc, zc, R = _taubin_sphere_fit_3d(points[:, 0], points[:, 1], points[:, 2])
        return np.array([xc, yc, zc, R]), R, zc
    xc, zc, R = _taubin_circle_fit_2d(points[:, 0], points[:, 2])
    return np.array([xc, zc, R]), R, zc


def _angle_from_cap(z_wall: float, zc: float, R: float) -> float | None:
    """Contact angle from ``cos θ = (z_wall - zc) / R`` or None if no intersection."""
    delta_z = z_wall - zc
    if abs(delta_z) >= R:
        return None
    return float(np.degrees(np.arccos(delta_z / R)))
