import numpy as np

from wetting_angle_kit.analysis.slicing.surface_definition import (
    SurfaceDefinition,
)
from wetting_angle_kit.io_utils import validate_droplet_geometry


class SlicingFrameFitter:
    """Slicing radial line method to estimate contact angle via circle fitting.

    Depending on ``droplet_geometry`` the droplet is analyzed by sweeping in y
    (cylinder modes) or by gamma azimuthal angle (spherical). For each slice / tilt
    a set of radial lines is sampled, a circle is fit to interface points, and
    the contact angle is derived from intersection with the lowest surface level.
    """

    #: Default azimuthal step (degrees) between radial sampling lines used
    #: by :class:`SurfaceDefinition` when building the per-slice interface.
    DEFAULT_DELTA_ANGLE = 8.0

    def __init__(
        self,
        liquid_coordinates: np.ndarray,
        max_dist: float,
        liquid_geom_center: np.ndarray,
        droplet_geometry: str = "spherical",
        delta_gamma: float | None = None,
        delta_cylinder: float | None = None,
        surface_filter_offset: float = 2.0,
        points_per_angstrom: float = 1.0,
        density_sigma: float = SurfaceDefinition.DEFAULT_DENSITY_SIGMA,
        delta_angle: float = DEFAULT_DELTA_ANGLE,
    ) -> None:
        """
        Parameters
        ----------
        liquid_coordinates : ndarray, shape (N, 3)
            Oxygen (or liquid marker) coordinates.
        max_dist : float
            Maximum radial distance for line sampling.
        liquid_geom_center : ndarray, shape (3,)
            Geometric droplet center; y component overridden per slice in cylinder
            modes.
        droplet_geometry : str, default 'spherical'
            One of ``{'spherical', 'cylinder_x', 'cylinder_y'}`` controlling slicing
            axis.
        delta_gamma : float, optional
            Angular step (degrees) for spherical droplet geometry
            (required if spherical).
        delta_cylinder : float, optional
            Step size along the slicing axis for cylindrical droplet geometry
            (required if cylinder_x / cylinder_y).
        surface_filter_offset : float, default 2.0
            Offset added to minimum droplet height for interface point filtering.
        points_per_angstrom : float, default 1.0
            Sampling density along each radial ray.
        density_sigma : float, default SurfaceDefinition.DEFAULT_DENSITY_SIGMA
            Gaussian smoothing width (Å) for the density-along-ray kernel.
        delta_angle : float, default DEFAULT_DELTA_ANGLE
            Azimuthal spacing (degrees) between radial lines.
        """
        validate_droplet_geometry(droplet_geometry)
        if droplet_geometry == "spherical":
            if delta_gamma is None:
                raise ValueError("delta_gamma must be provided for spherical analysis")
            if delta_cylinder is not None:
                raise ValueError(
                    "delta_cylinder must not be set for spherical analysis "
                    "(it is only valid for cylinder_x / cylinder_y)."
                )
        else:  # cylinder_x / cylinder_y
            if delta_cylinder is None:
                raise ValueError(
                    f"delta_cylinder must be provided for {droplet_geometry}."
                )
            if delta_gamma is not None:
                raise ValueError(
                    f"delta_gamma must not be set for {droplet_geometry} "
                    "(it is only valid for spherical)."
                )
        self.liquid_coordinates = liquid_coordinates
        self.max_dist = max_dist
        # Store a copy: predict_contact_angle mutates this in-place per slice
        # and we must not modify the caller's array.
        self.liquid_geom_center = np.array(liquid_geom_center, copy=True)
        self.droplet_geometry = droplet_geometry
        self.delta_gamma = delta_gamma
        self.delta_cylinder = delta_cylinder
        self.surface_filter_offset = surface_filter_offset
        # Sampling density along each radial ray; raise this (e.g. 2.0 or
        # higher) for small droplets where 1 sample per Å is insufficient
        # to fit the interface tanh profile.
        self.points_per_angstrom = points_per_angstrom
        # Gaussian smoothing width (Å) for the density-along-ray kernel and
        # azimuthal spacing (deg) between radial lines.
        # Tuned for the full atomistic model of liquid water
        # at room temperature by default; adjust for other liquids.
        self.density_sigma = density_sigma
        self.delta_angle = delta_angle

    def _slice_sweep(self) -> tuple[list[float], list[float]]:
        """Build the per-slice ``(axis_values, gammas)`` sweep once.

        Cylindrical mode sweeps the axial extent of ``liquid_coordinates``
        in ``delta_cylinder`` steps with ``gamma = 0``. Spherical mode
        repeats the droplet's y-center and rotates ``gamma`` from 0° to
        180° in ``delta_gamma`` steps. The two public list accessors
        below project this single source of truth.
        """
        if self.droplet_geometry in ("cylinder_y", "cylinder_x"):
            axis_values = self.liquid_coordinates[:, 1]
            ys = list(
                np.arange(
                    float(axis_values.min()),
                    float(axis_values.max()),
                    self.delta_cylinder,
                )
            )
            return ys, [0.0] * len(ys)
        if self.delta_gamma is None:
            raise ValueError("delta_gamma is required for droplet_geometry='spherical'")
        n_slices = int(180 / self.delta_gamma)
        gammas = list(np.linspace(0.0, 180.0, n_slices))
        return [float(self.liquid_geom_center[1])] * n_slices, gammas

    def calculate_y_axis_list(self) -> list[float]:
        """Return the per-slice center position along the slicing axis.

        Returns
        -------
        list[float]
            Y positions of slice centers; for spherical, the droplet center
            y is repeated ``180 / delta_gamma`` times.
        """
        return self._slice_sweep()[0]

    def calculate_gammas_list(self) -> list[float]:
        """Return the gamma tilt angle (degrees) for each slice."""
        return self._slice_sweep()[1]

    def surface_definition(self, v_gamma: float) -> tuple[np.ndarray, np.ndarray]:
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
        rr, xz = surface_def.analyze_lines()
        return np.array(xz), np.array(rr)

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

    @staticmethod
    def fit_circle(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Algebraic (Kasa) least-squares circle fit.

        Linearises ``(x - xc)^2 + (z - zc)^2 = R^2`` into
        ``2 xc·x + 2 zc·z + c = x^2 + z^2`` with ``c = R^2 - xc^2 - zc^2``,
        and solves the resulting overdetermined linear system in one
        ``np.linalg.lstsq`` call. Replaces the previous SciPy non-linear
        fit, which was the slicing hot path's main per-slice cost and
        which depended on a sensible initial guess.

        Parameters
        ----------
        x_data : ndarray
            X coordinates.
        y_data : ndarray
            Z coordinates.

        Returns
        -------
        ndarray, shape (3,)
            ``[x_center, z_center, radius]``.

        Raises
        ------
        np.linalg.LinAlgError
            If the input points are collinear (rank-deficient system).
        ValueError
            If the algebraic solution yields a non-positive squared radius
            (degenerate sample, e.g. all points on a line).
        """
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
        a_matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        rhs = x * x + y * y
        sol, _, _, _ = np.linalg.lstsq(a_matrix, rhs, rcond=None)
        xc, zc, c = float(sol[0]), float(sol[1]), float(sol[2])
        r_sq = c + xc * xc + zc * zc
        if r_sq <= 0.0:
            raise ValueError(
                f"Algebraic circle fit produced non-positive R^2 ({r_sq:.3g}); "
                "the surface points are likely degenerate."
            )
        return np.array([xc, zc, float(np.sqrt(r_sq))])

    def find_intersection(self, popt: np.ndarray, y_line: float) -> float | None:
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
            Contact angle in degrees or None if circle does not intersect baseline.
        """
        _, z_center, radius = popt
        delta_z = y_line - z_center
        discriminant = radius**2 - delta_z**2
        if discriminant < 0:
            return None
        theta = np.arccos(delta_z / radius)
        return float(np.degrees(theta))

    def predict_contact_angle(
        self,
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        """Run slicing loop and return per-slice contact angles and geometry.

        Only slices for which the full pipeline (surface detection, circle
        fit, and baseline intersection) succeeds contribute to the returned
        lists. The three lists are kept in lockstep: ``angles[i]``,
        ``surfaces[i]``, and ``popt_arrays[i]`` always describe the same
        slice, so that a single index can be used across all three.

        Returns
        -------
        tuple(list[float], list[np.ndarray], list[np.ndarray])
            (angles, surfaces, popt_arrays) where
            angles : list of contact angles (deg)
            surfaces : list of surface point arrays (each (M, 2))
            popt_arrays : list of fitted circle parameter arrays extended by
            baseline + offset
        """
        gammas = self.calculate_gammas_list()
        y_axis_list = self.calculate_y_axis_list()
        angles: list[float] = []
        surfaces: list[np.ndarray] = []
        popt_arrays: list[np.ndarray] = []
        for counter, value_gamma in enumerate(gammas):
            self.liquid_geom_center[1] = y_axis_list[counter]
            surf, rr = self.surface_definition(value_gamma)
            if surf.size == 0:
                continue
            min_drop = float(np.min(surf[:, 1]))
            limit_med = min_drop + self.surface_filter_offset
            surf_line = self.separate_surface_data(surf, limit_med)
            if len(surf_line) < 3:  # need at least 3 points to fit a circle
                continue
            x_data = surf_line[:, 0]
            y_data = surf_line[:, 1]
            try:
                popt = self.fit_circle(x_data, y_data)
            except (np.linalg.LinAlgError, ValueError):
                continue
            angle = self.find_intersection(popt, min_drop)
            if angle is None:
                continue
            angles.append(angle)
            surfaces.append(surf)
            popt_arrays.append(np.append(popt, limit_med))
        return angles, surfaces, popt_arrays
