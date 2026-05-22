import numpy as np

from wetting_angle_kit.contact_angle_methods.binning.surface_definition import (
    HyperbolicTangentModel,
)


def test_hyperbolic_tangent_compute_isoline_well_formed():
    """Wall inside the fitted sphere should yield finite isoline arrays."""
    model = HyperbolicTangentModel()
    # Wall at z=0, droplet center at z=8, radius 10 → wall is inside the
    # sphere (|0 - 8| = 8 < 10). Densities and thicknesses are positive.
    model.params = [1.0, 0.0, 10.0, 8.0, 0.0, 1.0, 1.0]
    circle_xi, circle_zi, wall_xi, wall_zi = model.compute_isoline()
    assert circle_xi.size == 100
    assert np.all(np.isfinite(circle_xi))
    assert np.all(np.isfinite(circle_zi))
    assert np.all(np.isfinite(wall_xi))
    np.testing.assert_allclose(wall_zi, 0.0)
