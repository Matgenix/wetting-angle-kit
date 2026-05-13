"""Interface models used by the binning analyzer.

Algorithm
---------

The implemented :class:`HyperbolicTangentModel` represents the
liquid–vapor interface as a product of two sigmoids,

::

    rho(xi, zi) = g(r) * h(z),
    g(r) = 0.5 * [(rho1 + rho2) - (rho1 - rho2) * tanh(2 (r - R_eq) / t1)],
    h(z) = 0.5 * [1 + tanh(2 * (zi - zi_0) / t2)],
    r    = sqrt(xi**2 + (zi - zi_c)**2),

with seven free parameters fitted by non-linear least squares:

* ``rho1`` — liquid-phase number density (particles · Å⁻³).
* ``rho2`` — vapor-phase number density (particles · Å⁻³).
* ``R_eq`` — equivalent spherical radius (Å).
* ``zi_c`` — z-coordinate of the spherical center (Å).
* ``zi_0`` — wall reference z-coordinate (Å).
* ``t1``   — radial interface thickness (Å).
* ``t2``   — vertical interface thickness (Å).

Bounds keep densities and lengths in their physical ranges. Once the fit
converges, the contact angle is the geometric tangent angle of the
fitted sphere at the wall intersection. Lengths are in Å, angles in
degrees.
"""

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


class SurfaceModel:
    """Abstract base for surface models used in contact angle analysis.

    Subclasses must implement ``fit`` and ``evaluate``.

    Parameters
    ----------
    initial_params : sequence of float, optional
        Initial guess for model parameters. Interpretation is left to subclasses.
    """

    def __init__(self, initial_params: list[float] | None = None) -> None:
        """Store initial parameters and prepare covariance placeholder."""
        self.params = initial_params
        self.covariance = None

    def fit(
        self, x_data: Any, density_data: np.ndarray
    ) -> "SurfaceModel":  # pragma: no cover - abstract
        """Fit the model to density data.

        Parameters
        ----------
        x_data : Any
            Coordinate representation consumed by the concrete model.
        density_data : ndarray
            1D array of density values matching ``x_data``.

        Returns
        -------
        SurfaceModel
            The fitted model instance (``self``) for chaining.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(self, x: Any) -> float:  # pragma: no cover - abstract
        """Evaluate the fitted function at point ``x``.

        Parameters
        ----------
        x : Any
            Coordinate(s) accepted by the concrete model.

        Returns
        -------
        float
            Evaluated density value.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate_on_grid(self, xi_grid: np.ndarray, zi_grid: np.ndarray) -> np.ndarray:
        """Evaluate the fitted function on a 2D (xi, zi) grid.

        Parameters
        ----------
        xi_grid : sequence of float
            Radial or in-plane coordinate values.
        zi_grid : sequence of float
            Height (z) coordinate values.

        Returns
        -------
        ndarray, shape (len(xi_grid), len(zi_grid))
            2D array of evaluated density values.
        """
        out_fitted = np.zeros((len(xi_grid), len(zi_grid)))
        for i in range(len(xi_grid)):
            for j in range(len(zi_grid)):
                out_fitted[i, j] = self.evaluate((xi_grid[i], zi_grid[j]))
        return out_fitted


class HyperbolicTangentModel(SurfaceModel):
    """Liquid–vapor interface model using a hyperbolic tangent profile.

    The density field is modeled as the product of two sigmoidal (tanh) terms: one
    depending on the spherical radial distance and one along the vertical axis.

    Parameters
    ----------
    initial_params : list[float], optional
        Seven parameters ``[rho1, rho2, R_eq, zi_c, zi_0, t1, t2]`` used as
        the starting guess for the non-linear fit. Defaults to
        :attr:`DEFAULT_INITIAL_PARAMS`, which is tuned for water at room
        temperature in Å units; supply system-specific values if your
        density or droplet size differs.

        - rho1 : Liquid-phase density.
        - rho2 : Vapor-phase density.
        - R_eq : Equivalent spherical radius.
        - zi_c : z-coordinate of the sphere center.
        - zi_0 : Reference wall z-coordinate.
        - t1 : Interface thickness (radial component).
        - t2 : Interface thickness (vertical component).
    """

    #: Default initial guess for the seven fit parameters (water at RT, Å units).
    DEFAULT_INITIAL_PARAMS = [1e-3, 3e-2, 40.0, 20.0, 4.0, 1.0, 1.0]

    def __init__(self, initial_params: list[float] | None = None) -> None:
        if initial_params is None:
            initial_params = list(self.DEFAULT_INITIAL_PARAMS)
        super().__init__(initial_params)

    def _fitting_function(
        self,
        x: Any,
        rho1: float,
        rho2: float,
        R_eq: float,
        zi_c: float,
        zi_0: float,
        t1: float,
        t2: float,
    ) -> Any:
        """Internal hyperbolic tangent density expression.

        Parameters
        ----------
        x : tuple(float, float)
            Coordinates ``(xi, zi)``.
        rho1, rho2 : float
            Liquid and vapor densities.
        R_eq : float
            Sphere radius.
        zi_c : float
            Sphere center z-coordinate.
        zi_0 : float
            Wall reference z-coordinate.
        t1, t2 : float
            Interface thickness parameters (radial, vertical).

        Returns
        -------
        float
            Density value at the given coordinates.
        """
        xi, zi = x[0], x[1]

        def g(r: Any) -> Any:
            return 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))

        def h(z: Any) -> Any:
            return 0.5 * (1 + np.tanh(2 * z / t2))

        r = np.sqrt(xi**2 + (zi - zi_c) ** 2)
        z = zi - zi_0
        return g(r) * h(z)

    # Physical bounds on the seven parameters
    # [rho1, rho2, R_eq, zi_c, zi_0, t1, t2].
    # Densities are non-negative, radius and interface thicknesses are
    # strictly positive. Center coordinates are unconstrained.
    _PARAM_LOWER = np.array([0.0, 0.0, 1e-6, -np.inf, -np.inf, 1e-6, 1e-6])
    _PARAM_UPPER = np.array([np.inf] * 7)

    def fit(self, x_data: Any, density_data: np.ndarray) -> "HyperbolicTangentModel":
        """Fit the model parameters to provided density samples.

        Parameters
        ----------
        x_data : tuple(ndarray, ndarray)
            Coordinate arrays ``(xi_array, zi_array)`` flattened or broadcastable.
        density_data : ndarray
            Density values corresponding to ``x_data``.

        Returns
        -------
        HyperbolicTangentModel
            Fitted model instance (``self``).
        """
        self.params, self.covariance = curve_fit(
            self._fitting_function,
            x_data,
            density_data,
            p0=self.params,
            bounds=(self._PARAM_LOWER, self._PARAM_UPPER),
            maxfev=1_000_000,
        )
        self._warn_if_at_bounds()
        return self

    def _warn_if_at_bounds(self) -> None:
        """Emit a warning if any fitted parameter is pinned at a finite bound.

        This usually indicates the hyperbolic tangent model is not a good
        fit (e.g. too few frames, wrong geometry, or noisy density field).
        """
        assert self.params is not None
        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        tol = 1e-6
        at_bound = []
        for name, value, lo, hi in zip(
            param_names, self.params, self._PARAM_LOWER, self._PARAM_UPPER, strict=False
        ):
            if np.isfinite(lo) and abs(value - lo) < tol * max(1.0, abs(lo)):
                at_bound.append(f"{name}={value:.3g} at lower bound {lo}")
            elif np.isfinite(hi) and abs(value - hi) < tol * max(1.0, abs(hi)):
                at_bound.append(f"{name}={value:.3g} at upper bound {hi}")
        if at_bound:
            warnings.warn(
                "Hyperbolic tangent fit converged with parameter(s) at the "
                "physical bound, suggesting a poor fit: " + "; ".join(at_bound),
                RuntimeWarning,
                stacklevel=3,
            )

    def evaluate(self, x: Any) -> float:
        """Evaluate the fitted hyperbolic tangent model at ``x``.

        Parameters
        ----------
        x : tuple(float, float)
            Coordinates ``(xi, zi)``.

        Returns
        -------
        float
            Density value at the given point.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before evaluation")
        return self._fitting_function(
            x,
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
            self.params[4],
            self.params[5],
            self.params[6],
        )

    def compute_isoline(
        self, scale_factor: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute an iso-surface circle and wall line approximation.

        Parameters
        ----------
        scale_factor : float, default 0.95
            Factor applied to fitted radius for visualization.

        Returns
        -------
        tuple(ndarray, ndarray, ndarray, ndarray)
            ``(circle_xi, circle_zi, wall_line_xi, wall_line_zi)`` arrays.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing isoline")

        R = scale_factor * self.params[2]  # R_eq
        Zcenter = self.params[3]  # zi_c
        Zwall = self.params[4]  # zi_0

        discriminant = R**2 - (Zwall - Zcenter) ** 2
        if discriminant < 0:
            raise ValueError(
                "Fitted wall is outside the fitted droplet radius "
                f"(R={R:.3f}, |Zwall - Zcenter|={abs(Zwall - Zcenter):.3f}); "
                "isoline cannot be computed. The hyperbolic tangent fit "
                "likely did not converge to a physical solution."
            )
        xi_wall = np.sqrt(discriminant)
        alpha_inf = np.arctan((Zwall - Zcenter) / xi_wall)
        alpha = np.linspace(alpha_inf, np.pi / 2, 100)

        Xicenter = 1.0
        circle_xi = Xicenter + R * np.cos(alpha)
        circle_zi = Zcenter + R * np.sin(alpha)

        wall_line_xi = np.linspace(Xicenter, xi_wall, 100)
        wall_line_zi = np.ones(len(wall_line_xi)) * Zwall

        return circle_xi, circle_zi, wall_line_xi, wall_line_zi

    def compute_contact_angle(self) -> float:
        """Return the contact angle (degrees) implied by fitted parameters.

        Returns
        -------
        float
            Contact angle in degrees. Returns ``nan`` if the fitted wall
            position lies outside the fitted droplet sphere (no
            intersection), which indicates a poor fit.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle")

        R_eq = self.params[2]
        zita_c = self.params[3]
        zita_wall = self.params[4]

        discriminant = R_eq**2 - (zita_wall - zita_c) ** 2
        if discriminant < 0:
            warnings.warn(
                "Fitted wall is outside the fitted droplet sphere "
                f"(R_eq={R_eq:.3f}, |zita_wall - zita_c|="
                f"{abs(zita_wall - zita_c):.3f}); contact angle is undefined.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        xi_cross = np.sqrt(discriminant)
        theta = (np.pi / 2 - np.arctan((zita_wall - zita_c) / xi_cross)) * 180 / np.pi
        return theta

    def get_parameters(self) -> dict[str, float]:
        """Return a mapping of parameter names to fitted values.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter names and values.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameters")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return {
            name: value for name, value in zip(param_names, self.params, strict=False)
        }

    def get_parameter_strings(self) -> list[str]:
        """Return formatted parameter strings suitable for logging.

        Returns
        -------
        list[str]
            Formatted parameter strings (``"name:value\\n"``).
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameter strings")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return [
            f"{name}:{value}\n"
            for name, value in zip(param_names, self.params, strict=False)
        ]
