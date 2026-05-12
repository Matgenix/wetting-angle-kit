from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from wetting_angle_kit.contact_angle_method.sliced_method.surface_defined import (
    SurfaceDefinition,
)
from wetting_angle_kit.io_utils import validate_droplet_geometry


class ContactAngleSliced:
    """Sliced radial line method to estimate contact angle via circle fitting.

    Depending on ``droplet_geometry`` the droplet is analyzed by sweeping in y
    (cylinder modes) or by gamma azimuthal angle (spherical). For each slice / tilt
    a set of radial lines is sampled, a circle is fit to interface points, and
    the contact angle is derived from intersection with the lowest surface
    level.

    Parameters
    ----------
    liquid_coordinates : ndarray, shape (N, 3)
        Oxygen (or liquid marker) coordinates.
    max_dist : float
        Maximum radial distance for line sampling.
    liquid_geom_center : ndarray, shape (3,)
        Geometric droplet center; y component overridden per slice in cylinder
        modes.
    droplet_geometry : str, default 'cylinder_y'
        One of {'cylinder_y', 'cylinder_x', 'spherical'} controlling slicing
        axis.
    delta_gamma : float, optional
        Angular increment (degrees) for spherical droplet geometry
        (required if spherical).
    width_cylinder : float, optional
        Extent in slicing axis direction (y or x) for cylindrical droplet geometry.
    delta_cylinder : float, optional
        Step size along slicing axis.
    surface_filter_offset : float, default 2.0
        Offset added to minimum droplet height for interface point filtering.
    """

    #: Default azimuthal step (degrees) between radial sampling lines used
    #: by :class:`SurfaceDefinition` when building the per-slice interface.
    DEFAULT_DELTA_ANGLE = 8.0

    def __init__(
        self,
        liquid_coordinates: np.ndarray,
        max_dist: float,
        liquid_geom_center: np.ndarray,
        droplet_geometry: str = "cylinder_y",
        delta_gamma: Optional[float] = None,
        width_cylinder: Optional[float] = None,
        delta_cylinder: Optional[float] = None,
        surface_filter_offset: float = 2.0,
        points_per_angstrom: float = 1.0,
        density_sigma: float = SurfaceDefinition.DEFAULT_DENSITY_SIGMA,
        delta_angle: float = DEFAULT_DELTA_ANGLE,
    ) -> None:
        validate_droplet_geometry(droplet_geometry)
        self.liquid_coordinates = liquid_coordinates
        self.max_dist = max_dist
        # Store a copy: predict_contact_angle mutates this in-place per slice
        # and we must not modify the caller's array.
        self.liquid_geom_center = np.array(liquid_geom_center, copy=True)
        self.droplet_geometry = droplet_geometry
        self.delta_gamma = delta_gamma
        self.width_cylinder = width_cylinder
        self.delta_cylinder = delta_cylinder
        self.surface_filter_offset = surface_filter_offset
        # Sampling density along each radial ray; raise this (e.g. 2.0 or
        # higher) for small droplets where 1 sample per Å is insufficient
        # to fit the interface tanh profile.
        self.points_per_angstrom = points_per_angstrom
        # Gaussian smoothing width (Å) for the density-along-ray kernel and
        # azimuthal spacing (deg) between radial lines. Tuned for water at
        # RT by default; adjust for other liquids.
        self.density_sigma = density_sigma
        self.delta_angle = delta_angle
        if self.droplet_geometry in ("cylinder_y", "cylinder_x") and (
            width_cylinder is None or delta_cylinder is None
        ):
            import warnings

            warnings.warn(
                "width_cylinder and delta_cylinder recommended for "
                f"{self.droplet_geometry}",
                UserWarning,
                stacklevel=2,
            )
        if self.droplet_geometry == "spherical" and delta_gamma is None:
            raise ValueError("delta_gamma must be provided for spherical analysis")

    def calculate_y_axis_list(self) -> List[float]:
        """Return axis position list for the chosen droplet geometry.

        For cylindrical droplets the slice positions sweep from 0 to
        ``width_cylinder`` in steps of ``delta_cylinder``. This assumes the
        simulation box origin is at 0 along the slicing axis (the LAMMPS
        convention). If your box uses a non-zero origin, supply
        ``liquid_geom_center`` already shifted into a 0-based frame, or
        pre-translate the trajectory before analysis.

        Returns
        -------
        list[float]
            Y (or X if 'cylinder_x') positions; spherical returns repeated center y.
        """
        if self.droplet_geometry in ("cylinder_y", "cylinder_x"):
            assert self.width_cylinder is not None and self.delta_cylinder is not None
            return list(np.arange(0, self.width_cylinder, self.delta_cylinder))
        if self.droplet_geometry == "spherical":
            assert self.delta_gamma is not None
            return [self.liquid_geom_center[1]] * int(180 / self.delta_gamma)
        return []

    def calculate_gammas_list(self) -> List[float]:
        """Return gamma inclination list for the chosen droplet geometry."""
        if self.droplet_geometry in ("cylinder_y", "cylinder_x"):
            assert self.width_cylinder is not None and self.delta_cylinder is not None
            return [
                0.0
                for _ in np.arange(
                    0,
                    self.width_cylinder,
                    self.delta_cylinder,
                )
            ]
        if self.droplet_geometry == "spherical":
            assert self.delta_gamma is not None
            return list(
                np.linspace(
                    0.0,
                    180.0,
                    int(180 / self.delta_gamma),
                )
            )
        return []

    def surface_definition(self, v_gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Sample interface lines for a given gamma.

        Parameters
        ----------
        v_gamma : float
            Gamma inclination in degrees (0 for cylindrical slices).

        Returns
        -------
        tuple(ndarray, ndarray)
            (surf_xz, radial_info); surf_xz (M,2), radial_info (M,2).
        """
        surface_def = SurfaceDefinition(
            self.liquid_coordinates,
            self.delta_angle,
            self.max_dist,
            self.liquid_geom_center,
            v_gamma,
            points_per_angstrom=self.points_per_angstrom,
            density_sigma=self.density_sigma,
        )
        list_rr, list_xz = surface_def.analyze_lines()
        return np.array(list_xz), np.array(list_rr)

    def separate_surface_data(self, surf: np.ndarray, limit_med: float) -> np.ndarray:
        """Filter surface points above reference height.

        Parameters
        ----------
        surf : ndarray, shape (M, 2)
            Surface XZ points.
        limit_med : float
            Baseline (minimum droplet height + offset).

        Returns
        -------
        ndarray
            Filtered subset of ``surf`` with z > ``limit_med``.
        """
        return surf[surf[:, 1] > limit_med]

    def fit_circle(
        self,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        initial_guess: Sequence[float],
    ) -> np.ndarray:
        """Perform non-linear least squares circle fit.

        Parameters
        ----------
        X_data : ndarray
            X coordinates.
        Y_data : ndarray
            Z coordinates.
        initial_guess : sequence
            Initial parameters [x_center, z_center, radius].

        Returns
        -------
        ndarray
            Optimized parameters [x_center, z_center, radius].
        """
        popt, _ = curve_fit(
            self.circle_equation,
            (X_data, Y_data),
            np.zeros_like(X_data),
            p0=initial_guess,
        )
        # The residual sqrt((x-xc)^2 + (z-zc)^2) - R is symmetric in the sign
        # of R, so curve_fit may converge to a negative radius. Normalize.
        popt[2] = float(abs(popt[2]))
        return popt

    def find_intersection(self, popt: np.ndarray, y_line: float) -> Optional[float]:
        """Compute contact angle from circle intersection with a baseline.

        Parameters
        ----------
        popt : sequence
            Circle parameters [x_center, z_center, radius].
        y_line : float
            Baseline z-coordinate (minimum droplet height).

        Returns
        -------
        float | None
            Contact angle (deg) or None if circle does not intersect baseline.
        """
        Xs, Ys, R = popt
        delta_y = y_line - Ys
        discriminant = R**2 - delta_y**2
        if discriminant < 0:
            return None
        theta = np.arccos(delta_y / R)
        return float(np.degrees(theta))

    def circle_equation(
        self,
        xy_data: Tuple[np.ndarray, np.ndarray],
        x_center: float,
        z_center: float,
        radius: float,
    ) -> np.ndarray:
        """Return residuals for circle equation used in fitting.

        Parameters
        ----------
        xy_data : tuple(ndarray, ndarray)
            (X_data, Y_data) coordinate arrays.
        x_center : float
            Circle center x.
        z_center : float
            Circle center z.
        radius : float
            Circle radius.

        Returns
        -------
        ndarray
            Residuals sqrt((x-xc)^2+(z-zc)^2) - R.
        """
        X_data, Y_data = xy_data
        return np.sqrt((X_data - x_center) ** 2 + (Y_data - z_center) ** 2) - radius

    def predict_contact_angle(
        self,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Run slicing loop and return per-slice contact angles and geometry.

        Only slices for which the full pipeline (surface detection, circle
        fit, and baseline intersection) succeeds contribute to the returned
        lists. The three lists are kept in lockstep: ``angles[i]``,
        ``surfaces[i]``, and ``popt_arrays[i]`` always describe the same
        slice, so consumers can use a single index (e.g. the median index)
        across all three.

        Returns
        -------
        tuple(list[float], list[ndarray], list[ndarray])
            (angles, surfaces, popt_arrays) where
            angles : list of contact angles (deg)
            surfaces : list of surface point arrays (each (M, 2))
            popt_arrays : list of fitted circle parameter arrays extended by
            baseline + offset
        """
        gammas = self.calculate_gammas_list()
        y_axis_list = self.calculate_y_axis_list()
        list_alfas: list[float] = []
        array_surfaces: list[np.ndarray] = []
        array_popt: list[np.ndarray] = []
        for counter, value_gamma in enumerate(gammas):
            self.liquid_geom_center[1] = y_axis_list[counter]
            surf, list_rr = self.surface_definition(value_gamma)
            if surf.size == 0:
                continue
            min_drop = float(np.min(surf[:, 1]))
            limit_med = min_drop + self.surface_filter_offset
            surf_line = self.separate_surface_data(surf, limit_med)
            if len(surf_line) < 3:  # need at least 3 points to fit a circle
                continue
            X_data = surf_line[:, 0]
            Y_data = surf_line[:, 1]
            mean_rr = (
                float(np.mean(list_rr[:, 0])) if list_rr.size else self.max_dist / 2
            )
            initial_guess = [
                self.liquid_geom_center[0],
                self.liquid_geom_center[2],
                mean_rr,
            ]
            try:
                popt = self.fit_circle(X_data, Y_data, initial_guess)
            except Exception:  # pragma: no cover - rare convergence failures
                continue
            angle = self.find_intersection(popt, min_drop)
            if angle is None:
                continue
            list_alfas.append(angle)
            array_surfaces.append(surf)
            array_popt.append(np.append(popt, limit_med))
        return list_alfas, array_surfaces, array_popt
