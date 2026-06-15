"""Shared density-on-rays kernel and batched tanh interface fit.

Used by the rays/grid extractors that fit a hyperbolic-tangent profile
to the density along a ray or sample it on a grid of cell centres.

:class:`GaussianDensityField` wraps a ``cKDTree`` over the atom cloud
plus the kernel-width parameters. :class:`HistogramDensityField` is
the equivalent for the histogram-style estimator. Both expose the
same ``evaluate(positions)`` method so the ray-fan geometry helpers
in :mod:`wetting_angle_kit.analysis.interface` and the
:class:`DensityEstimator` strategy can take either one.
:func:`fit_tanh_profiles_batched` solves the per-ray tanh fit for an
entire slice in one batched Gauss–Newton call.
"""

from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree


class DensityFieldProtocol(Protocol):
    """Density field used by the ray-fan extractors.

    Any object exposing :meth:`evaluate` mapping ``(M, 3)`` sample
    positions to an ``(M,)`` density array satisfies the protocol;
    both :class:`GaussianDensityField` and :class:`HistogramDensityField`
    are concrete implementations.
    """

    def evaluate(self, positions: np.ndarray) -> np.ndarray: ...


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


class HistogramDensityField:
    """Top-hat (histogram-style) density evaluator over a fixed atom cloud.

    The natural counterpart of :class:`GaussianDensityField` for the
    ``rays`` extractor with a binning density. Conceptually a 1D histogram of atoms
    projected onto each ray, implemented as a 3D top-hat kernel:
    each sample position counts atoms within a sphere of radius
    ``bin_width / 2`` and divides by the sphere's volume. This shares
    the ``cKDTree.sparse_distance_matrix`` machinery used by the
    Gaussian field and exposes the same ``evaluate`` interface so the
    ray-fan geometry helpers in :mod:`wetting_angle_kit.analysis.interface`
    can take either field.

    Parameters
    ----------
    atom_coords : ndarray, shape (N, 3)
        Atom positions used as the density sources.
    bin_width : float
        Diameter (Å) of the top-hat kernel — i.e. atoms within
        ``bin_width / 2`` of a sample position count, atoms outside
        do not. The natural analogue of ``density_sigma`` in the
        Gaussian field, but with a hard cutoff instead of a smooth
        fall-off.
    """

    def __init__(self, atom_coords: np.ndarray, bin_width: float) -> None:
        if bin_width <= 0.0:
            raise ValueError(f"bin_width must be positive; got {bin_width!r}.")
        self.bin_width = bin_width
        self._radius = bin_width / 2.0
        self._volume = (4.0 / 3.0) * np.pi * self._radius**3
        self._atom_tree: cKDTree | None = (
            cKDTree(atom_coords) if len(atom_coords) > 0 else None
        )

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Return per-sample atom-count density.

        For each sample position, counts the atoms inside a sphere of
        radius ``bin_width / 2`` and divides by that sphere's volume.

        Parameters
        ----------
        positions : ndarray, shape (M, 3)
            Sample coordinates.

        Returns
        -------
        ndarray, shape (M,)
            Density values (atoms / Å³) at each sample position.
        """
        n_samples = len(positions)
        if self._atom_tree is None or n_samples == 0:
            return np.zeros(n_samples)
        sample_tree = cKDTree(positions)
        pairs = sample_tree.sparse_distance_matrix(
            self._atom_tree, max_distance=self._radius, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.zeros(n_samples)
        counts = np.bincount(pairs["i"], minlength=n_samples).astype(float)
        return counts / self._volume


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
    damping converges in 3–6 iterations. The batched solve aborts the
    iteration if any ray's normal equations become singular (e.g. a
    near-constant density profile that never crosses the interface),
    leaving every ray at its current parameters; a non-finite update
    instead resets every ray to the closed-form initial guess. This
    early stop also regularises the noisier histogram density, whose
    fully converged per-ray optima can chase shot noise. The recovered
    ``zd`` is clipped to ``[0, distances[-1]]`` to keep ill-fit rays
    from escaping the sampling envelope.

    Parameters
    ----------
    distances : ndarray, shape (M,)
        Sample distances along the ray (same for every ray of a slice).
        Must be monotonically increasing; the last entry sets the
        clip bound on the recovered interface position.
    densities : ndarray, shape (R, M)
        Density values per ray.
    max_iter : int, default 25
        Hard cap on Gauss–Newton iterations.
    tol : float, default 1e-9
        Convergence threshold on the max absolute parameter step
        across all rays.

    Returns
    -------
    ndarray, shape (R,)
        Fitted ``zd`` (interface position) per ray, clipped to
        ``[0, distances[-1]]``.
    """
    z = np.ascontiguousarray(distances, dtype=np.float64)
    y = np.ascontiguousarray(densities, dtype=np.float64)
    n_rays, n_samples = y.shape
    max_dist = float(z[-1])

    rho_max = y.max(axis=1)
    rho_min = y.min(axis=1)
    h0 = 0.5 * (rho_max + rho_min)
    d0 = 0.5 * (rho_max - rho_min)
    zd0 = z[np.argmin(np.abs(y - h0[:, None]), axis=1)]
    zd0 = np.clip(zd0, 0.0, max_dist)
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
            # A degenerate ray (e.g. near-constant density) makes its
            # 3x3 system singular and aborts the whole batched solve.
            # Stop iterating and keep the current per-ray parameters;
            # for the dominant well-conditioned rays these have already
            # converged, and the early stop keeps noise-driven rays near
            # their robust closed-form seed.
            break
        params += step
        if not np.isfinite(params).all():
            params = params_init.copy()
            break
        if np.max(np.abs(step)) < tol:
            break

    return np.clip(params[:, 0], 0.0, max_dist)
