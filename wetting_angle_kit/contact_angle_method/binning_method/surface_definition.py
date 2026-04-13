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

    def __init__(self, initial_params=None):
        """Store initial parameters and prepare covariance placeholder."""
        self.params = initial_params
        self.covariance = None

    def fit(self, x_data, density_data):  # pragma: no cover - abstract
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

    def evaluate(self, x):  # pragma: no cover - abstract
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

    def evaluate_on_grid(self, xi_grid, zi_grid):
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
        Seven parameters ``[rho1, rho2, R_eq, zi_c, zi_0, t1, t2]``:

        - rho1 : Liquid-phase density.
        - rho2 : Vapor-phase density.
        - R_eq : Equivalent spherical radius.
        - zi_c : z-coordinate of the sphere center.
        - zi_0 : Reference wall z-coordinate.
        - t1 : Interface thickness (radial component).
        - t2 : Interface thickness (vertical component).
    """

    def __init__(self, initial_params=None):
        if initial_params is None:
            initial_params = [1e-3, 3e-2, 40.0, 20.0, 4.0, 1.0, 1.0]
        super().__init__(initial_params)

    def _fitting_function(self, x, rho1, rho2, R_eq, zi_c, zi_0, t1, t2):
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

        def g(r):
            return 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))

        def h(z):
            return 0.5 * (1 + np.tanh(2 * z / t2))

        r = np.sqrt(xi**2 + (zi - zi_c) ** 2)
        z = zi - zi_0
        return g(r) * h(z)

    def fit(self, x_data, density_data):
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
            maxfev=1_000_000,
        )
        return self

    def evaluate(self, x):
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

    def compute_isoline(self, scale_factor=0.95):
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

        xi_wall = np.sqrt(R**2 - (Zwall - Zcenter) ** 2)
        alpha_inf = np.arctan((Zwall - Zcenter) / xi_wall)
        alpha = np.linspace(alpha_inf, np.pi / 2, 100)

        Xicenter = 1.0
        circle_xi = Xicenter + R * np.cos(alpha)
        circle_zi = Zcenter + R * np.sin(alpha)

        wall_line_xi = np.linspace(Xicenter, xi_wall, 100)
        wall_line_zi = np.ones((len(wall_line_xi))) * Zwall

        return circle_xi, circle_zi, wall_line_xi, wall_line_zi

    def compute_contact_angle(self):
        """Return the contact angle (degrees) implied by fitted parameters.

        Returns
        -------
        float
            Contact angle in degrees.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle")

        R_eq = self.params[2]
        zita_c = self.params[3]
        zita_wall = self.params[4]

        xi_cross = np.sqrt(R_eq**2 - (zita_wall - zita_c) ** 2)
        theta = (np.pi / 2 - np.arctan((zita_wall - zita_c) / xi_cross)) * 180 / np.pi
        return theta

    def get_parameters(self):
        """Return a mapping of parameter names to fitted values.

        Returns
        -------
        dict[str, float]
            Dictionary of parameter names and values.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameters")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return {name: value for name, value in zip(param_names, self.params)}

    def get_parameter_strings(self):
        """Return formatted parameter strings suitable for logging.

        Returns
        -------
        list[str]
            Formatted parameter strings (``"name:value\\n"``).
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameter strings")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return [f"{name}:{value}\n" for name, value in zip(param_names, self.params)]
