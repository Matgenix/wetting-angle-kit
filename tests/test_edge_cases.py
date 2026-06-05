"""Edge-case tests for input validation, NaN guards, and deprecation paths."""

import numpy as np
import pytest

from wetting_angle_kit.analysis.binning.surface_definition import (
    HyperbolicTangentModel,
)
from wetting_angle_kit.analysis.slicing.angle_fitting import (
    SlicingFrameFitter,
)

# --- Invalid droplet_geometry should be rejected by both analyzers ---


def test_contact_angle_slicing_rejects_invalid_geometry():
    coords = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown droplet_geometry"):
        SlicingFrameFitter(
            liquid_coordinates=coords,
            max_dist=10,
            liquid_geom_center=np.zeros(3),
            droplet_geometry="not-a-real-geometry",
            delta_gamma=20,
        )


# --- Slicing predictor: empty result lists stay in lockstep ---


def test_predict_contact_angle_returns_aligned_lists():
    """Even if some slices fail, the three returned lists must have the same
    length. This guards against the historical bug where median_idx into
    angles would address a different slice in popt_arrays/surfaces."""
    coords = np.array([[0.0, 0.0, 10.0]])  # single atom = no tanh interface
    predictor = SlicingFrameFitter(
        liquid_coordinates=coords,
        max_dist=10,
        liquid_geom_center=np.zeros(3),
        droplet_geometry="spherical",
        delta_gamma=45,
    )
    angles, surfaces, popts = predictor.predict_contact_angle()
    assert len(angles) == len(surfaces) == len(popts)


def test_contact_angle_slicing_copies_geometric_center():
    """Constructor must not retain a reference to the caller's array."""
    center = np.array([1.0, 2.0, 3.0])
    predictor = SlicingFrameFitter(
        liquid_coordinates=np.zeros((1, 3)),
        max_dist=10,
        liquid_geom_center=center,
        droplet_geometry="spherical",
        delta_gamma=45,
    )
    predictor.liquid_geom_center[1] = 999.0
    # Caller's array must be untouched.
    np.testing.assert_array_equal(center, np.array([1.0, 2.0, 3.0]))


# --- Cylindrical mode without delta_cylinder raises ---


def test_slicing_cylinder_without_delta_cylinder_raises():
    with pytest.raises(ValueError, match="delta_cylinder"):
        SlicingFrameFitter(
            liquid_coordinates=np.zeros((3, 3)),
            max_dist=10,
            liquid_geom_center=np.zeros(3),
            droplet_geometry="cylinder_y",
        )


def test_slicing_spherical_requires_delta_gamma():
    with pytest.raises(ValueError, match="delta_gamma must be provided"):
        SlicingFrameFitter(
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


# --- BinningBatchFitter.get_profile_coordinates ---


def _make_binning_analyzer(parser):
    from wetting_angle_kit.analysis.binning import BinningBatchFitter

    return BinningBatchFitter(
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
    )


class _BoxedStubParser:
    """Helper that supplies the abstract box-size methods of ``BaseParser``.

    Subclasses only need to set ``frames`` (a list of ``(N, 3)`` arrays) and
    use the defaults below for a 100x100x100 orthogonal cell.
    """

    box: tuple[float, float, float] = (100.0, 100.0, 100.0)

    def box_size_x(self, frame_index):
        return self.box[0]

    def box_size_y(self, frame_index):
        return self.box[1]

    def box_length_max(self, frame_index):
        return max(self.box)


def test_binning_get_profile_coordinates_empty_frame_list():
    """Empty frame_indices must return empty arrays and zero frames."""
    from wetting_angle_kit.parsers.base import BaseParser

    class _StubParser(_BoxedStubParser, BaseParser):
        def parse(self, frame_index, indices=None):
            return np.zeros((0, 3))

        def frame_count(self):
            return 0

    analyzer = _make_binning_analyzer(_StubParser())
    r, z, n = analyzer.get_profile_coordinates(frame_indices=[])
    assert r.shape == (0,)
    assert z.shape == (0,)
    assert n == 0


def test_binning_get_profile_coordinates_concatenates_frames():
    """r and z arrays are concatenated across requested frames; z stays in lab frame."""
    from wetting_angle_kit.parsers.base import BaseParser

    frame0 = np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, 6.0], [0.0, 0.0, 7.0]])
    frame1 = np.array([[2.0, 0.0, 8.0], [-2.0, 0.0, 9.0], [0.0, 0.0, 10.0]])

    class _StubParser(_BoxedStubParser, BaseParser):
        # A large box so the per-frame circular mean coincides with the
        # arithmetic mean and the asserted radii do not depend on PBC
        # wrapping.
        def parse(self, frame_index, indices=None):
            return [frame0, frame1][frame_index]

        def frame_count(self):
            return 2

    analyzer = _make_binning_analyzer(_StubParser())
    r, z, n = analyzer.get_profile_coordinates(frame_indices=[0, 1])
    assert n == 2
    # Spherical r is non-negative and the per-frame center-of-mass projection
    # collapses pairs of mirror atoms to the same radius (1, 1, 0) and (2, 2, 0).
    np.testing.assert_allclose(r, np.array([1.0, 1.0, 0.0, 2.0, 2.0, 0.0]))
    # z is lab-frame, concatenated as-is.
    np.testing.assert_array_equal(z, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))


def test_binning_precentered_skips_box_probe():
    """``precentered=True`` must bypass the box probe entirely so the
    box-size accessors are never invoked, even by a parser that would raise
    if asked for box info."""
    from wetting_angle_kit.analysis.binning import BinningBatchFitter
    from wetting_angle_kit.parsers.base import BaseParser

    frame = np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, 6.0], [0.0, 0.0, 7.0]])

    class _NoBoxParser(BaseParser):
        def parse(self, frame_index, indices=None):
            return frame

        def frame_count(self):
            return 1

        def box_size_x(self, frame_index):
            raise AssertionError("box_size_x must not be called when precentered=True")

        def box_size_y(self, frame_index):
            raise AssertionError("box_size_y must not be called when precentered=True")

        def box_length_max(self, frame_index):
            raise AssertionError(
                "box_length_max must not be called when precentered=True"
            )

    analyzer = BinningBatchFitter(
        parser=_NoBoxParser(),
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
        precentered=True,
    )
    r, z, n = analyzer.get_profile_coordinates(frame_indices=[0])
    assert n == 1
    np.testing.assert_allclose(r, np.array([1.0, 1.0, 0.0]))
