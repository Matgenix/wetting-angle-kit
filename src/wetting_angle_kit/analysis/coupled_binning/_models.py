"""Hyperbolic-tangent models + heuristic-grid helpers for the coupled-binning analyzers.

Both the 2D (seven-parameter) and 3D (nine-parameter) joint density
models are kept in this module so the shared bounds / warning / cap-angle
formula sit side by side. Public access goes through
:class:`CoupledBinning2DAnalyzer` and :class:`CoupledBinning3DAnalyzer`.
"""

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

# Parameter names for the 2D and 3D models, used for at-bound warnings
# and for the public ``model_params`` dict on the batch result types.
_PARAM_NAMES = ("rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2")
_PARAM_NAMES_3D = (
    "rho1",
    "rho2",
    "R_eq",
    "xi_c",
    "yi_c",
    "zi_c",
    "zi_0",
    "t1",
    "t2",
)


# ----------------------------------------------------------------------
# 2D model.
# ----------------------------------------------------------------------


class _HyperbolicTangentModel2D:
    """Coupled 2D-binning joint contact-angle model.

    Density field modelled as a product of two sigmoidal (tanh) terms,
    one radial and one vertical:

    ::

        rho(xi, zi) = g(r) * h(zi - zi_0),
            g(r) = 0.5 * [(rho1 + rho2) - (rho1 - rho2) * tanh(2 (r - R_eq) / t1)],
            h(z) = 0.5 * [1 + tanh(2 z / t2)],
            r    = sqrt(xi^2 + (zi - zi_c)^2).

    Seven free parameters fitted by bounded NLLS. Private (the public
    entry point is :class:`CoupledBinning2DAnalyzer`); the 3D
    counterpart :class:`_HyperbolicTangentModel3D` lives in the same
    module.
    """

    DEFAULT_INITIAL_PARAMS = (1e-3, 3e-2, 40.0, 20.0, 4.0, 1.0, 1.0)

    _PARAM_LOWER = np.array([0.0, 0.0, 1e-6, -np.inf, -np.inf, 1e-6, 1e-6])
    _PARAM_UPPER = np.array([np.inf] * 7)

    def __init__(self, initial_params: list[float] | None = None) -> None:
        if initial_params is None:
            initial_params = list(self.DEFAULT_INITIAL_PARAMS)
        self.params: list[float] | np.ndarray | None = initial_params
        self.covariance: np.ndarray | None = None

    @staticmethod
    def _fitting_function(
        x: tuple[np.ndarray, np.ndarray],
        rho1: float,
        rho2: float,
        R_eq: float,
        zi_c: float,
        zi_0: float,
        t1: float,
        t2: float,
    ) -> np.ndarray:
        xi, zi = x[0], x[1]
        r = np.sqrt(xi**2 + (zi - zi_c) ** 2)
        g_r = 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
        h_z = 0.5 * (1.0 + np.tanh(2 * (zi - zi_0) / t2))
        return g_r * h_z

    def fit(
        self,
        x_data: tuple[np.ndarray, np.ndarray],
        density_data: np.ndarray,
    ) -> "_HyperbolicTangentModel2D":
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
        if self.params is None:
            return
        tol = 1e-6
        at_bound = []
        for name, value, lo, hi in zip(
            _PARAM_NAMES,
            self.params,
            self._PARAM_LOWER,
            self._PARAM_UPPER,
            strict=False,
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

    def compute_contact_angle(self) -> float:
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle.")
        R_eq = float(self.params[2])
        zi_c = float(self.params[3])
        zi_0 = float(self.params[4])
        discriminant = R_eq**2 - (zi_0 - zi_c) ** 2
        if discriminant < 0:
            warnings.warn(
                "Fitted wall is outside the fitted droplet sphere "
                f"(R_eq={R_eq:.3f}, |zi_0 - zi_c|={abs(zi_0 - zi_c):.3f}); "
                "contact angle is undefined.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        xi_cross = np.sqrt(discriminant)
        return float((np.pi / 2 - np.arctan((zi_0 - zi_c) / xi_cross)) * 180 / np.pi)


# ----------------------------------------------------------------------
# 3D model.
# ----------------------------------------------------------------------


class _HyperbolicTangentModel3D:
    """3D extension of the binning method's hyperbolic-tangent model.

    Density factorises into a radial sigmoid centred at ``(xi_c, yi_c,
    zi_c)`` and a vertical sigmoid above the wall ``zi_0``:

    ::

        rho(xi, yi, zi) = g(r) * h(zi - zi_0),
            g(r) = 0.5 * [(rho1 + rho2) - (rho1 - rho2) * tanh(2 (r - R_eq) / t1)],
            h(z) = 0.5 * [1 + tanh(2 z / t2)],
            r    = sqrt((xi - xi_c)^2 + (yi - yi_c)^2 + (zi - zi_c)^2).

    Bounds keep densities and lengths in their physical ranges, same as
    the 2D model. ``xi_c`` / ``yi_c`` carry the only extra degrees of
    freedom over the 2D fit.
    """

    #: Initial guess tuned for room-temperature water; the two
    #: horizontal centres default to ``0`` because the analyzer
    #: pre-centers the atoms on the droplet COM before binning.
    DEFAULT_INITIAL_PARAMS = (1e-3, 3e-2, 40.0, 0.0, 0.0, 20.0, 4.0, 1.0, 1.0)

    # Bounds vector order matches DEFAULT_INITIAL_PARAMS.
    _PARAM_LOWER = np.array(
        [0.0, 0.0, 1e-6, -np.inf, -np.inf, -np.inf, -np.inf, 1e-6, 1e-6]
    )
    _PARAM_UPPER = np.array([np.inf] * 9)

    def __init__(self, initial_params: list[float] | None = None) -> None:
        if initial_params is None:
            initial_params = list(self.DEFAULT_INITIAL_PARAMS)
        self.params: list[float] | np.ndarray | None = initial_params
        self.covariance: np.ndarray | None = None

    @staticmethod
    def _fitting_function(
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        rho1: float,
        rho2: float,
        R_eq: float,
        xi_c: float,
        yi_c: float,
        zi_c: float,
        zi_0: float,
        t1: float,
        t2: float,
    ) -> np.ndarray:
        xi, yi, zi = x[0], x[1], x[2]
        r = np.sqrt((xi - xi_c) ** 2 + (yi - yi_c) ** 2 + (zi - zi_c) ** 2)
        g_r = 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
        h_z = 0.5 * (1.0 + np.tanh(2 * (zi - zi_0) / t2))
        return g_r * h_z

    def fit(
        self,
        x_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        density_data: np.ndarray,
    ) -> "_HyperbolicTangentModel3D":
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
        if self.params is None:
            return
        tol = 1e-6
        at_bound = []
        for name, value, lo, hi in zip(
            _PARAM_NAMES_3D,
            self.params,
            self._PARAM_LOWER,
            self._PARAM_UPPER,
            strict=False,
        ):
            if np.isfinite(lo) and abs(value - lo) < tol * max(1.0, abs(lo)):
                at_bound.append(f"{name}={value:.3g} at lower bound {lo}")
            elif np.isfinite(hi) and abs(value - hi) < tol * max(1.0, abs(hi)):
                at_bound.append(f"{name}={value:.3g} at upper bound {hi}")
        if at_bound:
            warnings.warn(
                "3D hyperbolic tangent fit converged with parameter(s) at "
                "the physical bound, suggesting a poor fit: " + "; ".join(at_bound),
                RuntimeWarning,
                stacklevel=3,
            )

    def compute_contact_angle(self) -> float:
        """Return the contact angle (degrees) implied by the fitted parameters.

        Same geometric formula as the 2D model: the sphere of radius
        ``R_eq`` centred at ``(xi_c, yi_c, zi_c)`` intersects the wall
        plane ``z = zi_0`` in a circle whose tangent makes the contact
        angle with the wall.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle.")
        R_eq = float(self.params[2])
        zi_c = float(self.params[5])
        zi_0 = float(self.params[6])
        discriminant = R_eq**2 - (zi_0 - zi_c) ** 2
        if discriminant < 0:
            warnings.warn(
                "3D fit wall is outside the fitted droplet sphere "
                f"(R_eq={R_eq:.3f}, |zi_0 - zi_c|="
                f"{abs(zi_0 - zi_c):.3f}); contact angle is undefined.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        xi_cross = np.sqrt(discriminant)
        return float((np.pi / 2 - np.arctan((zi_0 - zi_c) / xi_cross)) * 180 / np.pi)


# ----------------------------------------------------------------------
# Heuristic binning grids.
# ----------------------------------------------------------------------


def _heuristic_binning_params(parser: Any) -> dict[str, Any]:
    """Build the legacy heuristic binning grid: 50×50 cells over a third
    of the largest in-plane box dimension.
    """
    max_dist = int(
        np.max(
            np.array(
                [
                    parser.box_size_y(frame_index=0),
                    parser.box_size_x(frame_index=0),
                ]
            )
        )
        / 3
    )
    warnings.warn(
        "binning_params was not supplied; using a heuristic default "
        f"(xi_0=0, xi_f={max_dist}, zi_0=0, zi_f={max_dist}, 50x50 bins) "
        "derived from one third of the largest in-plane box dimension. "
        "For accurate density fields, supply system-specific "
        "binning_params matching your droplet size and per-frame sampling.",
        UserWarning,
        stacklevel=3,
    )
    return {
        "xi_0": 0,
        "xi_f": max_dist,
        "nbins_xi": 50,
        "zi_0": 0.0,
        "zi_f": max_dist,
        "nbins_zi": 50,
    }


def _heuristic_binning_params_3d(parser: Any) -> dict[str, Any]:
    """Build a heuristic 3D binning grid centred on the droplet COM.

    Same one-third-of-box heuristic as the 2D version but tripled
    along all three axes. Emits a warning because the user almost
    always wants to override this.
    """
    half = int(
        np.max(
            np.array(
                [
                    parser.box_size_y(frame_index=0),
                    parser.box_size_x(frame_index=0),
                ]
            )
        )
        / 6
    )
    warnings.warn(
        "binning_params was not supplied; using a heuristic default "
        f"(xi/yi in [-{half}, {half}], zi in [0, {2 * half}], 30^3 bins). "
        "For accurate density fields, supply system-specific "
        "binning_params matching your droplet size and per-frame sampling.",
        UserWarning,
        stacklevel=3,
    )
    return {
        "xi_0": -half,
        "xi_f": half,
        "nbins_xi": 30,
        "yi_0": -half,
        "yi_f": half,
        "nbins_yi": 30,
        "zi_0": 0.0,
        "zi_f": 2 * half,
        "nbins_zi": 30,
    }
