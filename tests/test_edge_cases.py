"""Edge-case tests for input validation, NaN guards, and deprecation paths."""

from __future__ import annotations

import numpy as np
import pytest

from wetting_angle_kit.contact_angle_methods.binning.surface_definition import (
    HyperbolicTangentModel,
)
from wetting_angle_kit.contact_angle_methods.sliced.angle_fitting import (
    ContactAngleSliced,
)

# --- Invalid droplet_geometry should be rejected by both analyzers ---


def test_contact_angle_sliced_rejects_invalid_geometry():
    coords = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown droplet_geometry"):
        ContactAngleSliced(
            liquid_coordinates=coords,
            max_dist=10,
            liquid_geom_center=np.zeros(3),
            droplet_geometry="not-a-real-geometry",
            delta_gamma=20,
        )


# --- Sliced predictor: empty result lists stay in lockstep ---


def test_predict_contact_angle_returns_aligned_lists():
    """Even if some slices fail, the three returned lists must have the same
    length. This guards against the historical bug where median_idx into
    angles would address a different slice in popt_arrays/surfaces."""
    coords = np.array([[0.0, 0.0, 10.0]])  # single atom = no tanh interface
    predictor = ContactAngleSliced(
        liquid_coordinates=coords,
        max_dist=10,
        liquid_geom_center=np.zeros(3),
        droplet_geometry="spherical",
        delta_gamma=45,
    )
    angles, surfaces, popts = predictor.predict_contact_angle()
    assert len(angles) == len(surfaces) == len(popts)


def test_contact_angle_sliced_copies_geometric_center():
    """Constructor must not retain a reference to the caller's array."""
    center = np.array([1.0, 2.0, 3.0])
    predictor = ContactAngleSliced(
        liquid_coordinates=np.zeros((1, 3)),
        max_dist=10,
        liquid_geom_center=center,
        droplet_geometry="spherical",
        delta_gamma=45,
    )
    predictor.liquid_geom_center[1] = 999.0
    # Caller's array must be untouched.
    np.testing.assert_array_equal(center, np.array([1.0, 2.0, 3.0]))


# --- Cylindrical mode without delta_cylinder/width_cylinder warns ---


def test_sliced_cylinder_without_width_warns():
    with pytest.warns(UserWarning, match="width_cylinder and delta_cylinder"):
        ContactAngleSliced(
            liquid_coordinates=np.zeros((3, 3)),
            max_dist=10,
            liquid_geom_center=np.zeros(3),
            droplet_geometry="cylinder_y",
        )


def test_sliced_spherical_requires_delta_gamma():
    with pytest.raises(ValueError, match="delta_gamma must be provided"):
        ContactAngleSliced(
            liquid_coordinates=np.zeros((3, 3)),
            max_dist=10,
            liquid_geom_center=np.zeros(3),
            droplet_geometry="spherical",
        )


# --- HyperbolicTangentModel ---


def test_hyperbolic_tangent_requires_fit_before_use():
    model = HyperbolicTangentModel()
    # params is the initial guess (not None), so evaluate works, but
    # computing the contact angle / isoline requires the params to come
    # from a real fit. We at least verify the path explicitly:
    model.params = None
    with pytest.raises(ValueError, match="must be fitted"):
        model.compute_contact_angle()
    with pytest.raises(ValueError, match="must be fitted"):
        model.compute_isoline()
    with pytest.raises(ValueError, match="must be fitted"):
        model.evaluate((0.0, 0.0))


def test_hyperbolic_tangent_compute_contact_angle_nan_for_unphysical_fit():
    """When the wall sits outside the fitted sphere, the analyzer should
    return NaN rather than crash."""
    model = HyperbolicTangentModel()
    # rho1, rho2, R_eq, zi_c, zi_0, t1, t2 — wall far below center, R small.
    model.params = [1.0, 0.0, 5.0, 10.0, -50.0, 1.0, 1.0]
    with pytest.warns(RuntimeWarning, match="wall is outside"):
        angle = model.compute_contact_angle()
    assert np.isnan(angle)


def test_hyperbolic_tangent_compute_isoline_raises_for_unphysical_fit():
    model = HyperbolicTangentModel()
    model.params = [1.0, 0.0, 5.0, 10.0, -50.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="wall is outside"):
        model.compute_isoline()


# --- Factory rejects unknown methods ---


def test_contact_angle_analyzer_factory_rejects_unknown_method(tmp_path):
    from wetting_angle_kit.contact_angle_methods import contact_angle_analyzer

    with pytest.raises(ValueError, match="Unknown method"):
        contact_angle_analyzer(
            method="not-a-method",
            parser=object(),
            output_dir=str(tmp_path),
        )


# --- ContactAngleBinning.get_profile_coordinates ---


def _make_binning_analyzer(parser, tmp_path):
    from wetting_angle_kit.contact_angle_methods.binning import ContactAngleBinning

    return ContactAngleBinning(
        parser=parser,
        atom_indices=None,
        droplet_geometry="spherical",
        binning_params={
            "xi_0": 0.0,
            "xi_f": 10.0,
            "nbins_xi": 5,
            "zi_0": 0.0,
            "zi_f": 10.0,
            "nbins_zi": 5,
        },
        output_dir=str(tmp_path),
        plot_graphs=False,
    )


def test_binning_get_profile_coordinates_empty_frame_list(tmp_path):
    """Empty frame_indices must return empty arrays and zero frames."""
    from wetting_angle_kit.parsers.base import BaseParser

    class _StubParser(BaseParser):
        def parse(self, frame_index, indices=None):
            return np.zeros((0, 3))

        def frame_count(self):
            return 0

    analyzer = _make_binning_analyzer(_StubParser(), tmp_path)
    r, z, n = analyzer.get_profile_coordinates(frame_indices=[])
    assert r.shape == (0,)
    assert z.shape == (0,)
    assert n == 0


def test_binning_get_profile_coordinates_concatenates_frames(tmp_path):
    """r and z arrays are concatenated across requested frames; z stays in lab frame."""
    from wetting_angle_kit.parsers.base import BaseParser

    frame0 = np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, 6.0], [0.0, 0.0, 7.0]])
    frame1 = np.array([[2.0, 0.0, 8.0], [-2.0, 0.0, 9.0], [0.0, 0.0, 10.0]])

    class _StubParser(BaseParser):
        def parse(self, frame_index, indices=None):
            return [frame0, frame1][frame_index]

        def frame_count(self):
            return 2

    analyzer = _make_binning_analyzer(_StubParser(), tmp_path)
    r, z, n = analyzer.get_profile_coordinates(frame_indices=[0, 1])
    assert n == 2
    # Spherical r is non-negative and the per-frame center-of-mass projection
    # collapses pairs of mirror atoms to the same radius (1, 1, 0) and (2, 2, 0).
    np.testing.assert_allclose(r, np.array([1.0, 1.0, 0.0, 2.0, 2.0, 0.0]))
    # z is lab-frame, concatenated as-is.
    np.testing.assert_array_equal(z, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
