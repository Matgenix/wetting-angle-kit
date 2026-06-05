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

The density evaluator and the batched tanh fit are imported from
:mod:`wetting_angle_kit.analysis._density`, which is also used by the
new ray-based interface extractors. This module only adds the
slice-specific :meth:`SurfaceDefinition.analyze_lines` orchestration
on top.

All lengths are expected in Ångströms; angles are in degrees.
"""

import numpy as np

from wetting_angle_kit.analysis._density import (
    DEFAULT_CUTOFF_SIGMA as _DEFAULT_CUTOFF_SIGMA,
)
from wetting_angle_kit.analysis._density import (
    DEFAULT_DENSITY_SIGMA as _DEFAULT_DENSITY_SIGMA,
)
from wetting_angle_kit.analysis._density import (
    MIN_POINTS_PER_RAY as _MIN_POINTS_PER_RAY,
)
from wetting_angle_kit.analysis._density import (
    GaussianDensityField,
    fit_tanh_profiles_batched,
    tanh_profile,
)


class SurfaceDefinition:
    """Radial line sampling interface estimator for slicing contact angle.

    For each attitudinal angle beta the density is sampled along a ray emerging
    from the droplet geometric center. A simple tanh profile is fitted to obtain
    the interface position ("re") which is then projected back to XZ plane.
    """

    # Re-exported class attributes for callers that read e.g.
    # ``SurfaceDefinition.DEFAULT_DENSITY_SIGMA`` (notably
    # ``SlicingFrameFitter.__init__`` in this package). The canonical
    # values live in :mod:`wetting_angle_kit.analysis._density`.
    MIN_POINTS_PER_RAY = _MIN_POINTS_PER_RAY
    DEFAULT_DENSITY_SIGMA = _DEFAULT_DENSITY_SIGMA
    DEFAULT_CUTOFF_SIGMA = _DEFAULT_CUTOFF_SIGMA

    def __init__(
        self,
        atom_coords: np.ndarray,
        delta_angle: float,
        max_dist: float,
        center_geom: np.ndarray,
        gamma: float,
        density_conversion: float = 1.0,
        points_per_angstrom: float = 1.0,
        density_sigma: float = _DEFAULT_DENSITY_SIGMA,
        cutoff_sigma: float = _DEFAULT_CUTOFF_SIGMA,
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
        # Density evaluator now shared with the new ray-based
        # extractors via ``analysis/_density.py``. Empty atom clouds
        # are handled inside the field (it short-circuits to zeros).
        self._density = GaussianDensityField(
            atom_coords=atom_coords,
            density_sigma=density_sigma,
            cutoff_sigma=cutoff_sigma,
        )

    def density_contribution(self, positions: np.ndarray) -> np.ndarray:
        """Return Gaussian-smoothed density contributions at sample positions.

        Delegates to :meth:`GaussianDensityField.evaluate`; kept as a
        method for backwards compatibility with code that wraps
        ``SurfaceDefinition``.

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
        return self._density.evaluate(positions)

    @staticmethod
    def density_profile(z: np.ndarray, zd: float, d: float, h: float) -> np.ndarray:
        """Hyperbolic-tangent liquid–vapor density profile.

        Thin wrapper around
        :func:`wetting_angle_kit.analysis._density.tanh_profile`; kept
        as a static method for backwards compatibility.

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
        return tanh_profile(z, zd, d, h)

    def _fit_density_profiles_batched(
        self,
        distances: np.ndarray,
        densities: np.ndarray,
        *,
        max_iter: int = 25,
        tol: float = 1e-9,
    ) -> np.ndarray:
        """Fit ``rho(s) = d * tanh(zd - s) + h`` to every ray of a slice at once.

        Delegates to
        :func:`wetting_angle_kit.analysis._density.fit_tanh_profiles_batched`;
        this method exists to preserve the per-instance ``self.max_dist``
        binding for in-process callers.

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
        return fit_tanh_profiles_batched(
            distances,
            densities,
            max_dist=self.max_dist,
            max_iter=max_iter,
            tol=tol,
        )

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
