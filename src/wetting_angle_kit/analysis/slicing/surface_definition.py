"""Slicing-method interface estimator.

Algorithm
---------

For a single droplet slice the interface is recovered in two steps:

1. **Radial line scan.** A fan of rays is emitted from the droplet
   geometric center in the slice plane, with one ray every
   ``delta_angle`` degrees. Along each ray we evaluate a
   3D-Gaussian-smoothed density at uniformly spaced sampling points
   (``points_per_angstrom`` per Å, with a hard minimum of
   ``MIN_POINTS_PER_RAY``). The Gaussian kernel width
   ``density_sigma`` (Å) defaults to 3.0 Å, tuned for the full atomistic model of
   liquid water at room temperature.
2. **Interface fit.** A hyperbolic tangent profile
   ``rho(s) = d * tanh(zd - s) + h`` is fitted to the density along
   the ray, where ``s`` is the distance from the center (Å). The
   fitted ``zd`` is the interface position; the corresponding (x, z)
   point in the slice plane is returned.

All lengths are expected in Ångströms; angles are in degrees.
"""

import numpy as np
from scipy.spatial import cKDTree


class SurfaceDefinition:
    """Radial line sampling interface estimator for slicing contact angle.

    For each attitudinal angle beta the density is sampled along a ray emerging
    from the droplet geometric center. A simple tanh profile is fitted to obtain
    the interface position ("re") which is then projected back to XZ plane.
    """

    # Minimum number of sampling points along each ray. Below this the
    # tanh profile fit becomes numerically unreliable.
    MIN_POINTS_PER_RAY = 20

    # Default Gaussian standard deviation (Å) for the density-along-ray
    # smoothing kernel. Tuned for the full atomistic model of water at room temperature;
    # larger values broaden contributions and smooth the interface.
    DEFAULT_DENSITY_SIGMA = 3.0

    # Per-atom truncation radius for the Gaussian kernel, in units of
    # ``density_sigma``. At 5 sigma each excluded atom contributes
    # exp(-12.5) ≈ 3.7e-6 of the peak per-atom density: well below the
    # noise of a single-frame fit, while shrinking the inner kernel sum
    # from O(N) to the active neighbourhood of each sample point.
    DEFAULT_CUTOFF_SIGMA = 5.0

    def __init__(
        self,
        atom_coords: np.ndarray,
        delta_angle: float,
        max_dist: float,
        center_geom: np.ndarray,
        gamma: float,
        density_conversion: float = 1.0,
        points_per_angstrom: float = 1.0,
        density_sigma: float = DEFAULT_DENSITY_SIGMA,
        cutoff_sigma: float = DEFAULT_CUTOFF_SIGMA,
    ) -> None:
        """
        Parameters
        ----------
        atom_coords : ndarray, shape (N, 3)
            Cartesian coordinates of liquid atoms.
        delta_angle : float
            Angular step (degrees) between successive sampling rays.
        max_dist : float
            Maximum radial distance sampled along each ray.
        center_geom : ndarray, shape (3,)
            Approximate droplet geometric center.
        gamma : float
            Tilt angle (degrees) controlling rotation about the x-axis.
        density_conversion : float, default 1.0
            Factor applied multiplicatively to raw density contributions.
        points_per_angstrom : float, default 1.0
            Sampling density along each ray.
        density_sigma : float, default DEFAULT_DENSITY_SIGMA
            Gaussian kernel width (Å) for density smoothing.
        cutoff_sigma : float, default DEFAULT_CUTOFF_SIGMA
            Multiple of ``density_sigma`` beyond which atoms are excluded
            from each sample's density sum. Set higher for stricter
            agreement with the dense kernel; the cost grows roughly as
            ``cutoff_sigma ** 3`` (volume of the neighbour sphere).
        """
        self.atom_coords = atom_coords
        self.center_geom = center_geom
        self.density_conversion = density_conversion
        self.gamma = gamma
        self.delta_angle = delta_angle
        self.max_dist = max_dist
        self.points_per_angstrom = points_per_angstrom
        self.density_sigma = density_sigma
        self.cutoff_sigma = cutoff_sigma
        # Spatial index over the atomic coordinates so each ray's density
        # sum touches only the active neighbourhood of every sample point
        # instead of the O(M*N) broadcast that previously dominated the
        # slicing hot path. None for the empty-input case, which causes
        # density_contribution to short-circuit to zeros.
        self._atom_tree: cKDTree | None = (
            cKDTree(atom_coords) if len(atom_coords) > 0 else None
        )

    def density_contribution(self, positions: np.ndarray) -> np.ndarray:
        """Return Gaussian-smoothed density contributions at sample positions.

        Atoms farther than ``cutoff_sigma * density_sigma`` from a sample
        point are skipped; their kernel weight is below ~4e-6 of the peak
        at the 5 sigma default. Every (sample, atom) pair within the cutoff
        is enumerated in a single C-side call via
        ``cKDTree.sparse_distance_matrix`` so the per-sample work happens in
        one vectorised numpy pass instead of an M-iteration Python loop.

        Parameters
        ----------
        positions : ndarray, shape (M, 3)
            Ray sampling coordinates. ``M`` is typically the sample count of
            one ray, or the stacked count of all rays of a slice when
            :meth:`analyze_lines` batches the per-slice fan.

        Returns
        -------
        ndarray, shape (M,)
            Density values at each sampling position.
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

    @staticmethod
    def density_profile(z: np.ndarray, zd: float, d: float, h: float) -> np.ndarray:
        """Simple hyperbolic tangent profile used for liquid-vapor interface
        localization.

        Parameters
        ----------
        z : ndarray
            Distances along the sampling ray (Å).
        zd : float
            Liquid-vapor interface position parameter to be fitted.
        d : float
            Amplitude scaling parameter.
        h : float
            Offset parameter.

        Returns
        -------
        ndarray
            Modeled density values at each z.
        """
        return np.tanh(-z + zd) * d + h

    def _fit_density_profiles_batched(
        self,
        distances: np.ndarray,
        densities: np.ndarray,
        *,
        max_iter: int = 25,
        tol: float = 1e-9,
    ) -> np.ndarray:
        """Fit ``rho(s) = d * tanh(zd - s) + h`` to every ray of a slice at once.

        All rays of the slice share the same sampling grid, so the
        Jacobian's structure is identical across rays and the per-ray
        normal equations are independent 3x3 systems. A batched
        Gauss-Newton solver assembles those systems on numpy tensors and
        calls ``np.linalg.solve`` once per iteration, replacing the
        per-ray ``scipy.optimize.curve_fit`` (TRF + finite-difference
        Jacobian) that dominated the slicing hot path after 4.1/4.2.

        The closed-form initial guess (``h ~ midpoint``, ``d ~
        half-amplitude``, ``zd ~ midpoint crossing``) seeds each ray in
        the basin of the global minimum, so plain Gauss-Newton without
        damping converges in 3–6 iterations. Rays whose normal equations
        become singular (e.g. constant density) fall back to that
        initial guess.

        Parameters
        ----------
        distances : ndarray, shape (M,)
            Sample distances along the ray (same for every ray of a slice).
        densities : ndarray, shape (R, M)
            Density values per ray.
        max_iter : int, default 25
            Hard cap on Gauss-Newton iterations.
        tol : float, default 1e-9
            Convergence threshold on the max absolute parameter step
            across all rays.

        Returns
        -------
        ndarray, shape (R,)
            Fitted ``zd`` (interface position) per ray, clipped into
            ``[0, max_dist]`` to match the bounded behaviour of the
            original per-ray fit.
        """
        z = np.ascontiguousarray(distances, dtype=np.float64)
        y = np.ascontiguousarray(densities, dtype=np.float64)
        n_rays, n_samples = y.shape

        rho_max = y.max(axis=1)
        rho_min = y.min(axis=1)
        h0 = 0.5 * (rho_max + rho_min)
        d0 = 0.5 * (rho_max - rho_min)
        zd0 = z[np.argmin(np.abs(y - h0[:, None]), axis=1)]
        zd0 = np.clip(zd0, 0.0, float(self.max_dist))
        params = np.stack([zd0, d0, h0], axis=1)
        params_init = params.copy()

        for _ in range(max_iter):
            zd = params[:, 0]
            d = params[:, 1]
            h = params[:, 2]
            # u = tanh(zd - z), shape (R, M).
            u = np.tanh(zd[:, None] - z[None, :])
            residuals = y - (d[:, None] * u + h[:, None])
            # J columns are d/dzd, d/dd, d/dh. J_h = 1 is folded into the
            # normal equations directly (sums / counts), so only J_zd and
            # J_d are materialised here.
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

        return np.clip(params[:, 0], 0.0, float(self.max_dist))

    def analyze_lines(self) -> tuple[list[list[float]], list[list[float]]]:
        """Sample density along radial lines and fit interface positions.

        All rays of the slice share the same sampling distances and the
        same atomic neighbourhood, so their sample positions are stacked
        into a single ``(R * M, 3)`` array and the truncated density is
        evaluated in one ``density_contribution`` call. Only the tanh fit
        and the (x, z) projection are still done per ray.

        Returns
        -------
        rr : list[list[float]]
            Fitted interface distances and azimuth angles ``[interface_re, beta_deg]``.
        xz : list[list[float]]
            Projected interface coordinates ``[x_proj, z_proj]`` in XZ plane.
        """
        beta = np.linspace(0, 360, int(360 / self.delta_angle), endpoint=False)
        n_samples = max(
            int(self.max_dist * self.points_per_angstrom), self.MIN_POINTS_PER_RAY
        )
        cos_beta = np.cos(np.deg2rad(beta))
        sin_beta = np.sin(np.deg2rad(beta))
        cos_gamma = np.cos(np.deg2rad(self.gamma))
        sin_gamma = np.sin(np.deg2rad(self.gamma))

        # Per-ray unit direction vectors, shape (R, 3). Matches the original
        # per-iteration construction ``(cos_beta * cos_gamma,
        # cos_beta * sin_gamma, sin_beta)``.
        directions = np.column_stack(
            (cos_beta * cos_gamma, cos_beta * sin_gamma, sin_beta)
        )
        distances = np.linspace(0.0, self.max_dist, n_samples)

        # positions[r, m, :] = center_geom + distances[m] * directions[r, :]
        positions_rm = (
            self.center_geom[None, None, :]
            + distances[None, :, None] * directions[:, None, :]
        )
        density_flat = self.density_contribution(positions_rm.reshape(-1, 3))
        densities = self.density_conversion * density_flat.reshape(len(beta), n_samples)
        interface_re = self._fit_density_profiles_batched(distances, densities)

        x_proj = cos_beta * interface_re + self.center_geom[0]
        z_proj = sin_beta * interface_re + self.center_geom[2]
        rr = [[float(interface_re[i]), float(beta[i])] for i in range(len(beta))]
        xz = [[float(x_proj[i]), float(z_proj[i])] for i in range(len(beta))]
        return rr, xz
