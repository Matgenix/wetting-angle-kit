import warnings

import numpy as np
import pytest

from wetting_angle_kit.analysis.binning.surface_definition import (
    HyperbolicTangentModel,
)

# Reference parameter set used across the analytic checks below.
# Wall at z=0 sits inside a sphere of radius 10 centered at z=8.
_REF_PARAMS = [1.0, 0.0, 10.0, 8.0, 0.0, 1.0, 1.0]
_PARAM_NAMES = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]


def _fitted_model(params=_REF_PARAMS) -> HyperbolicTangentModel:
    model = HyperbolicTangentModel()
    model.params = list(params)
    return model


# --- compute_isoline -----------------------------------------------------


def test_hyperbolic_tangent_compute_isoline_well_formed():
    """Wall inside the fitted sphere should yield finite isoline arrays
    whose points exactly satisfy the scaled-sphere and wall equations."""
    model = _fitted_model()
    circle_xi, circle_zi, wall_xi, wall_zi = model.compute_isoline()
    assert circle_xi.size == 100
    assert wall_xi.size == 100

    # Circle points sit on the visualization sphere of radius
    # scale_factor * R_eq centered at (0, z_center).
    r = 0.95 * _REF_PARAMS[2]  # scale_factor * R_eq
    z_center = _REF_PARAMS[3]
    np.testing.assert_allclose(circle_xi**2 + (circle_zi - z_center) ** 2, r**2)
    # The contact point closes the arc at xi = sqrt(r^2 - (z_wall - z_c)^2),
    # z = z_wall; the arc ends at the sphere apex (xi=0, z=z_c+r).
    z_wall = _REF_PARAMS[4]
    xi_contact = np.sqrt(r**2 - (z_wall - z_center) ** 2)
    assert circle_xi[0] == pytest.approx(xi_contact)
    assert circle_zi[0] == pytest.approx(z_wall)
    assert circle_xi[-1] == pytest.approx(0.0, abs=1e-12)
    assert circle_zi[-1] == pytest.approx(z_center + r)

    # Wall line spans [0, xi_contact] at constant z = z_wall.
    np.testing.assert_allclose(wall_zi, z_wall)
    assert wall_xi[0] == pytest.approx(0.0)
    assert wall_xi[-1] == pytest.approx(xi_contact)


def test_compute_isoline_raises_when_wall_outside_sphere():
    # |z_wall - z_center| = 12 > R_eq = 10 → no intersection → ValueError.
    model = _fitted_model([1.0, 0.0, 10.0, 0.0, 12.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="outside the fitted droplet radius"):
        model.compute_isoline()


def test_compute_isoline_requires_fit_first():
    model = HyperbolicTangentModel()
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.compute_isoline()


# --- compute_contact_angle ------------------------------------------------


def test_compute_contact_angle_wall_at_equator_is_ninety_degrees():
    # Sphere center on the wall (zi_c = zi_0) → tangent at intersection is
    # vertical → contact angle is 90°.
    model = _fitted_model([1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0])
    assert model.compute_contact_angle() == pytest.approx(90.0)


def test_compute_contact_angle_wall_above_center_gives_acute_angle():
    # zi_0 - zi_c = +5, R_eq = 10 → xi_cross = sqrt(75); contact angle 60°.
    model = _fitted_model([1.0, 0.0, 10.0, 0.0, 5.0, 1.0, 1.0])
    assert model.compute_contact_angle() == pytest.approx(60.0)


def test_compute_contact_angle_wall_below_center_gives_obtuse_angle():
    # zi_0 - zi_c = -5, R_eq = 10 → droplet sits past its equator on the
    # wall → contact angle 120°.
    model = _fitted_model([1.0, 0.0, 10.0, 5.0, 0.0, 1.0, 1.0])
    assert model.compute_contact_angle() == pytest.approx(120.0)


def test_compute_contact_angle_returns_nan_when_wall_outside_sphere():
    model = _fitted_model([1.0, 0.0, 10.0, 0.0, 12.0, 1.0, 1.0])
    with pytest.warns(RuntimeWarning, match="outside the fitted droplet sphere"):
        angle = model.compute_contact_angle()
    assert np.isnan(angle)


def test_compute_contact_angle_requires_fit_first():
    model = HyperbolicTangentModel()
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.compute_contact_angle()


# --- evaluate / evaluate_on_grid -----------------------------------------


def test_evaluate_matches_fitting_function():
    model = _fitted_model()
    xi, zi = 3.0, 4.0
    rho1, rho2, R_eq, zi_c, zi_0, t1, t2 = _REF_PARAMS
    r = np.sqrt(xi**2 + (zi - zi_c) ** 2)
    z = zi - zi_0
    expected = (
        0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
    ) * (0.5 * (1 + np.tanh(2 * z / t2)))
    assert model.evaluate((xi, zi)) == pytest.approx(expected)


def test_evaluate_requires_fit_first():
    model = HyperbolicTangentModel()
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.evaluate((0.0, 0.0))


def test_evaluate_on_grid_shape_and_values():
    model = _fitted_model()
    xi_grid = np.array([0.0, 1.0, 2.0, 3.0])
    zi_grid = np.array([4.0, 5.0])
    grid = model.evaluate_on_grid(xi_grid, zi_grid)
    assert grid.shape == (len(xi_grid), len(zi_grid))
    # Spot-check entries against scalar evaluate calls (indexing='ij').
    for i, xi in enumerate(xi_grid):
        for j, zi in enumerate(zi_grid):
            assert grid[i, j] == pytest.approx(model.evaluate((xi, zi)))


# --- get_parameters / get_parameter_strings ------------------------------


def test_get_parameters_maps_names_to_values():
    model = _fitted_model()
    params = model.get_parameters()
    assert list(params.keys()) == _PARAM_NAMES
    assert list(params.values()) == _REF_PARAMS


def test_get_parameters_requires_fit_first():
    model = HyperbolicTangentModel()
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.get_parameters()


def test_get_parameter_strings_format():
    model = _fitted_model()
    strings = model.get_parameter_strings()
    assert len(strings) == len(_PARAM_NAMES)
    for name, value, line in zip(_PARAM_NAMES, _REF_PARAMS, strings, strict=True):
        assert line == f"{name}:{value}\n"


def test_get_parameter_strings_requires_fit_first():
    model = HyperbolicTangentModel()
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.get_parameter_strings()


# --- fit (round-trip on a synthetic density field) -----------------------


def test_fit_recovers_synthetic_parameters():
    # Matches the call style used by BinningBatchFitter: flattened
    # (xi, zi) coordinates and a flattened density vector.
    true_params = [0.02, 0.001, 12.0, 6.0, 0.0, 1.5, 1.2]
    xi_grid = np.linspace(0.1, 25.0, 30)
    zi_grid = np.linspace(-5.0, 25.0, 35)
    xi_mesh, zi_mesh = np.meshgrid(xi_grid, zi_grid, indexing="ij")
    xi_flat = xi_mesh.ravel()
    zi_flat = zi_mesh.ravel()

    seed_model = HyperbolicTangentModel(initial_params=list(true_params))
    truth = seed_model._fitting_function((xi_flat, zi_flat), *true_params)

    # Start from a perturbed initial guess to make the recovery non-trivial.
    perturbed = [p * 1.1 for p in true_params]
    model = HyperbolicTangentModel(initial_params=perturbed)
    fitted = model.fit((xi_flat, zi_flat), truth)
    assert fitted is model
    np.testing.assert_allclose(model.params, true_params, rtol=1e-4, atol=1e-4)


def test_warn_if_at_bounds_fires_when_parameter_pinned():
    # Drive ``_warn_if_at_bounds`` directly: the TRF solver inside ``fit``
    # keeps iterates strictly feasible, so it's hard to land exactly on a
    # bound through curve_fit. The warning logic itself is what matters.
    model = HyperbolicTangentModel()
    # t1 sits at its lower bound of 1e-6.
    model.params = np.array([1e-3, 1e-3, 10.0, 0.0, 0.0, 1e-6, 1.0])
    with pytest.warns(RuntimeWarning, match="at the physical bound"):
        model._warn_if_at_bounds()


def test_warn_if_at_bounds_silent_when_parameters_interior():
    # Interior values across all seven parameters; _REF_PARAMS itself has
    # rho2=0 sitting on its lower bound and would (correctly) warn.
    model = HyperbolicTangentModel()
    model.params = np.array([1e-3, 1e-3, 10.0, 0.0, 0.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        model._warn_if_at_bounds()
