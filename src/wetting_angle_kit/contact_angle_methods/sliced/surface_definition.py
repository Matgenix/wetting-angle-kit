"""Sliced-method interface estimator.

Algorithm
---------

For a single droplet slice the interface is recovered in two steps:

1. **Radial line scan.** A fan of rays is emitted from the droplet
   geometric center in the slice plane, with one ray every
   ``delta_angle`` degrees. Along each ray we evaluate a
   3D-Gaussian-smoothed density at uniformly spaced sampling points
   (``points_per_angstrom`` per Å, with a hard minimum of
   ``MIN_POINTS_PER_RAY``). The Gaussian kernel width
   ``density_sigma`` (Å) defaults to 3.0 Å, tuned for liquid water at
   room temperature.
2. **Interface fit.** A hyperbolic tangent profile
   ``rho(s) = d * tanh(zd - s) + h`` is fitted to the density along
   the ray, where ``s`` is the distance from the center (Å). The
   fitted ``zd`` is the interface position; the corresponding (x, z)
   point in the slice plane is returned.

All lengths are expected in Ångströms; angles are in degrees.
"""

import numpy as np
from scipy.optimize import curve_fit


class SurfaceDefinition:
    """Radial line sampling interface estimator for sliced contact angle.

    For each attitudinal angle beta the density is sampled along a ray emerging
    from the droplet geometric center. A simple tanh profile is fitted to obtain
    the interface position ("re") which is then projected back to XZ plane.
    """

    #: Minimum number of sampling points along each ray. Below this the
    #: tanh profile fit becomes numerically unreliable.
    MIN_POINTS_PER_RAY = 20

    #: Default Gaussian standard deviation (Å) for the density-along-ray
    #: smoothing kernel. Tuned for water at room temperature; larger values
    #: broaden contributions and smooth the interface.
    DEFAULT_DENSITY_SIGMA = 3.0

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
    ) -> None:
        """
        Parameters
        ----------
        atom_coords : ndarray, shape (N, 3)
            Cartesian coordinates of liquid atoms.
        delta_angle : float
            Angular increment (degrees) between successive sampling rays.
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
        """
        self.atom_coords = atom_coords
        self.center_geom = center_geom
        self.density_conversion = density_conversion
        self.gamma = gamma
        self.delta_angle = delta_angle
        self.max_dist = max_dist
        self.points_per_angstrom = points_per_angstrom
        self.density_sigma = density_sigma

    @staticmethod
    def density_contribution(
        positions: np.ndarray, coords: np.ndarray, sigma: float = 2.0
    ) -> np.ndarray:
        """Return Gaussian-smoothed density contributions at sampling positions.

        Parameters
        ----------
        positions : ndarray, shape (M, 3)
            Ray sampling coordinates.
        coords : ndarray, shape (N, 3)
            Atom coordinates contributing to density.
        sigma : float, default 2.0
            Gaussian standard deviation (Å). Larger values broaden contributions.

        Returns
        -------
        ndarray, shape (M,)
            Density values at each sampling position.
        """
        sigma2 = sigma * sigma
        prefactor = 1.0 / (2 * np.pi * sigma2) ** 1.5
        differences = positions[:, np.newaxis, :] - coords[np.newaxis, :, :]
        ri2 = np.sum(differences**2, axis=-1)
        den_contributions = prefactor * np.exp(-ri2 / (2 * sigma2))
        return np.sum(den_contributions, axis=1)

    @staticmethod
    def density_profile(z: np.ndarray, zd: float, d: float, h: float) -> np.ndarray:
        """Simple hyperbolic tangent profile used for interface localization.

        Parameters
        ----------
        z : ndarray
            Distances along the sampling ray (Å).
        zd : float
            Interface position parameter to be fitted.
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

    def fit_density_profile(
        self,
        z_data: np.ndarray,
        density: np.ndarray,
        param_bounds: tuple[list[float], list[float]],
    ) -> float:
        """Fit the profile and return estimated interface position.

        Parameters
        ----------
        z_data : ndarray
            Distances along the ray.
        density : ndarray
            Observed (smoothed) density values.
        param_bounds : tuple(list, list)
            Lower and upper bounds for ``(zd, d, h)``.

        Returns
        -------
        float
            Fitted ``zd`` value (interface location).
        """
        popt, _ = curve_fit(self.density_profile, z_data, density, bounds=param_bounds)
        zd, d, h = popt  # noqa: F841 - d, h retained for clarity if extended later
        return zd

    def analyze_lines(self) -> tuple[list[list[float]], list[list[float]]]:
        """Sample density along radial lines and fit interface positions.

        Returns
        -------
        rr : list[list[float]]
            Fitted interface distances and azimuth angles ``[interface_re, beta_deg]``.
        xz : list[list[float]]
            Projected interface coordinates ``[x_proj, z_proj]`` in XZ plane.
        """
        beta = np.linspace(0, 360, int(360 / self.delta_angle), endpoint=False)
        rr = []
        xz = []
        nn = max(int(self.max_dist * self.points_per_angstrom), self.MIN_POINTS_PER_RAY)
        param_bounds = ([0.0, -10.0, -10.0], [self.max_dist, 10.0, 10.0])
        cos_beta = np.cos(np.deg2rad(beta))
        sin_beta = np.sin(np.deg2rad(beta))
        cos_gamma = np.cos(np.deg2rad(self.gamma))
        sin_gamma = np.sin(np.deg2rad(self.gamma))
        for i in range(len(beta)):
            x_dir = cos_beta[i] * cos_gamma
            y_dir = sin_gamma * cos_beta[i]
            z_dir = sin_beta[i]
            direction = np.array([x_dir, y_dir, z_dir])
            positions = np.linspace(
                self.center_geom,
                self.center_geom + self.max_dist * direction,
                int(nn),
            )
            distances = np.linspace(0.0, self.max_dist, int(nn))
            density = self.density_conversion * self.density_contribution(
                positions,
                self.atom_coords,
                sigma=self.density_sigma,
            )
            interface_re = self.fit_density_profile(distances, density, param_bounds)
            rr.append([interface_re, beta[i]])
            xz.append(
                [
                    cos_beta[i] * interface_re + self.center_geom[0],
                    sin_beta[i] * interface_re + self.center_geom[2],
                ]
            )
        return rr, xz
