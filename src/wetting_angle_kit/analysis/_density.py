"""Shared density-on-rays kernel and batched tanh interface fit.

Used by both the slicing method (re-imported by
:class:`wetting_angle_kit.analysis.slicing.surface_definition.SurfaceDefinition`)
and the new ``rays_gaussian`` / ``rays_binning`` extractors that fit
a hyperbolic-tangent profile to the density along a ray.

:class:`GaussianDensityField` wraps a ``cKDTree`` over the atom cloud
plus the kernel-width parameters. :func:`fit_tanh_profiles_batched`
solves the per-ray tanh fit for an entire slice in one batched
Gauss–Newton call.
"""

import numpy as np
from scipy.spatial import cKDTree

#: Minimum number of sampling points along each ray. Below this the
#: tanh profile fit becomes numerically unreliable.
MIN_POINTS_PER_RAY = 20

#: Default Gaussian standard deviation (Å) for the density-along-ray
#: smoothing kernel. Tuned for the full atomistic model of water at
#: room temperature; larger values broaden contributions and smooth
#: the interface.
DEFAULT_DENSITY_SIGMA = 3.0

#: Per-atom truncation radius for the Gaussian kernel, in units of
#: ``density_sigma``. At 5 sigma each excluded atom contributes
#: ``exp(-12.5) ≈ 3.7e-6`` of the peak per-atom density: well below
#: the noise of a single-frame fit, while shrinking the inner kernel
#: sum from ``O(N)`` to the active neighbourhood of each sample point.
DEFAULT_CUTOFF_SIGMA = 5.0


class GaussianDensityField:
    """Truncated-Gaussian density evaluator over a fixed atom cloud.

    Parameters
    ----------
    atom_coords : ndarray, shape (N, 3)
        Atom positions used as the density sources.
    density_sigma : float, default :data:`DEFAULT_DENSITY_SIGMA`
        Gaussian kernel standard deviation (Å).
    cutoff_sigma : float, default :data:`DEFAULT_CUTOFF_SIGMA`
        Per-atom kernel truncation in units of ``density_sigma``.
    """

    def __init__(
        self,
        atom_coords: np.ndarray,
        density_sigma: float = DEFAULT_DENSITY_SIGMA,
        cutoff_sigma: float = DEFAULT_CUTOFF_SIGMA,
    ) -> None:
        self.density_sigma = density_sigma
        self.cutoff_sigma = cutoff_sigma
        # cKDTree over the atomic coordinates so each sample point's
        # density touches only the active neighbourhood instead of the
        # full N atoms. ``None`` for the empty-input case; ``evaluate``
        # short-circuits to a zeros array in that branch.
        self._atom_tree: cKDTree | None = (
            cKDTree(atom_coords) if len(atom_coords) > 0 else None
        )

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Return Gaussian-smoothed density at each sample position.

        Atoms farther than ``cutoff_sigma * density_sigma`` from a
        sample point are skipped; their kernel weight is below ~4e-6
        of the peak at the 5 sigma default. Every (sample, atom) pair
        within the cutoff is enumerated in a single C-side call via
        :meth:`scipy.spatial.cKDTree.sparse_distance_matrix`.

        Parameters
        ----------
        positions : ndarray, shape (M, 3)
            Sample coordinates.

        Returns
        -------
        ndarray, shape (M,)
            Density values at each sample position.
        """
        n_samples = len(positions)
        if self._atom_tree is None or n_samples == 0:
            return np.zeros(n_samples)
        sigma2 = self.density_sigma * self.density_sigma
        prefactor = 1.0 / (2 * np.pi * sigma2) ** 1.5
        cutoff = self.cutoff_sigma * self.density_sigma
        sample_tree = cKDTree(positions)
        pairs = sample_tree.sparse_distance_matrix(
            self._atom_tree, max_distance=cutoff, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.zeros(n_samples)
        contribs = prefactor * np.exp(-(pairs["v"] ** 2) / (2.0 * sigma2))
        return np.bincount(pairs["i"], weights=contribs, minlength=n_samples)


def tanh_profile(z: np.ndarray, zd: float, d: float, h: float) -> np.ndarray:
    """Hyperbolic-tangent liquid–vapor density profile.

    Parameters
    ----------
    z : ndarray
        Distances along the ray (Å).
    zd : float
        Interface position parameter to be fitted.
    d : float
        Amplitude scaling parameter (half the peak-to-vapor density
        range).
    h : float
        Vertical offset parameter (mid-point of liquid and vapor
        densities).

    Returns
    -------
    ndarray
        Modeled density values at each ``z``.
    """
    return np.tanh(-z + zd) * d + h


def fit_tanh_profiles_batched(
    distances: np.ndarray,
    densities: np.ndarray,
    *,
    max_dist: float,
    max_iter: int = 25,
    tol: float = 1e-9,
) -> np.ndarray:
    """Fit ``rho(s) = d * tanh(zd - s) + h`` to every ray of a slice at once.

    All rays of a slice share the same sampling grid, so the Jacobian
    structure is identical across rays and the per-ray normal equations
    are independent 3x3 systems. A batched Gauss–Newton solver
    assembles those systems on numpy tensors and calls
    :func:`numpy.linalg.solve` once per iteration — much faster than
    per-ray :func:`scipy.optimize.curve_fit`.

    The closed-form initial guess (``h ~ midpoint``, ``d ~
    half-amplitude``, ``zd ~ midpoint crossing``) seeds each ray in
    the basin of the global minimum, so plain Gauss–Newton without
    damping converges in 3–6 iterations. Rays whose normal equations
    become singular (e.g. constant density) fall back to the initial
    guess.

    Parameters
    ----------
    distances : ndarray, shape (M,)
        Sample distances along the ray (same for every ray of a slice).
    densities : ndarray, shape (R, M)
        Density values per ray.
    max_dist : float
        Upper bound on the fitted interface position; the returned
        ``zd`` is clipped to ``[0, max_dist]`` to keep ill-fit rays
        from escaping the sampling envelope.
    max_iter : int, default 25
        Hard cap on Gauss–Newton iterations.
    tol : float, default 1e-9
        Convergence threshold on the max absolute parameter step
        across all rays.

    Returns
    -------
    ndarray, shape (R,)
        Fitted ``zd`` (interface position) per ray, clipped to
        ``[0, max_dist]``.
    """
    z = np.ascontiguousarray(distances, dtype=np.float64)
    y = np.ascontiguousarray(densities, dtype=np.float64)
    n_rays, n_samples = y.shape

    rho_max = y.max(axis=1)
    rho_min = y.min(axis=1)
    h0 = 0.5 * (rho_max + rho_min)
    d0 = 0.5 * (rho_max - rho_min)
    zd0 = z[np.argmin(np.abs(y - h0[:, None]), axis=1)]
    zd0 = np.clip(zd0, 0.0, float(max_dist))
    params = np.stack([zd0, d0, h0], axis=1)
    params_init = params.copy()

    for _ in range(max_iter):
        zd = params[:, 0]
        d = params[:, 1]
        h = params[:, 2]
        # u = tanh(zd - z), shape (R, M).
        u = np.tanh(zd[:, None] - z[None, :])
        residuals = y - (d[:, None] * u + h[:, None])
        # J columns are d/dzd, d/dd, d/dh. J_h = 1 is folded into
        # the normal equations directly (sums / counts), so only
        # J_zd and J_d are materialised here.
        j_zd = d[:, None] * (1.0 - u * u)
        j_d = u
        # Symmetric 3x3 normal-equations matrix per ray.
        normal = np.empty((n_rays, 3, 3))
        normal[:, 0, 0] = np.einsum("rm,rm->r", j_zd, j_zd)
        normal[:, 0, 1] = normal[:, 1, 0] = np.einsum("rm,rm->r", j_zd, j_d)
        normal[:, 0, 2] = normal[:, 2, 0] = j_zd.sum(axis=1)
        normal[:, 1, 1] = np.einsum("rm,rm->r", j_d, j_d)
        normal[:, 1, 2] = normal[:, 2, 1] = j_d.sum(axis=1)
        normal[:, 2, 2] = n_samples
        rhs = np.empty((n_rays, 3))
        rhs[:, 0] = np.einsum("rm,rm->r", j_zd, residuals)
        rhs[:, 1] = np.einsum("rm,rm->r", j_d, residuals)
        rhs[:, 2] = residuals.sum(axis=1)
        try:
            # ``solve`` interprets the last two axes of the RHS as
            # ``(M, K)`` for batched LHS, so feed it a trailing K=1
            # axis to keep each ray's RHS a 3-vector.
            step = np.linalg.solve(normal, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        params += step
        if not np.isfinite(params).all():
            params = params_init.copy()
            break
        if np.max(np.abs(step)) < tol:
            break

    return np.clip(params[:, 0], 0.0, float(max_dist))
