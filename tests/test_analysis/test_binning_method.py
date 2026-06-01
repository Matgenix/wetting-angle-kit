import pathlib

import numpy as np
import pytest

from wetting_angle_kit.analysis import BinningTrajectoryAnalyzer
from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder


# --- Fixtures ---
@pytest.fixture
def filename():
    # Use the correct path for your test file
    return (
        pathlib.Path(__file__).parent.parent
        / "trajectories"
        / "traj_10_3_330w_nve_4k_reajust.lammpstrj"
    )


@pytest.fixture
def wat_find(filename):
    return LammpsDumpWaterFinder(
        filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2
    )


@pytest.fixture
def oxygen_indices(wat_find):
    return wat_find.get_water_oxygen_ids(0)


@pytest.fixture
def parser(filename):
    return LammpsDumpParser(filename)


@pytest.fixture
def binning_params():
    return {
        "xi_0": 0,
        "xi_f": 100.0,
        "nbins_xi": 50,
        "zi_0": 0.0,
        "zi_f": 100.0,
        "nbins_zi": 25,
    }


# --- Unit Test for BinningTrajectoryAnalyzer ---
@pytest.mark.integration
def test_binning_contact_angle_analyzer_with_real_data(
    filename, oxygen_indices, binning_params
):
    analyzer = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        width_cylinder=21,
        binning_params=binning_params,
    )

    results = analyzer.analyze([1])

    assert len(results) == 1
    # Cylindrical droplet on a graphene-like surface gives a contact angle
    # around 90-100° here. Use a moderate band so the test catches gross
    # regressions but tolerates the inherent noise of a single-frame fit.
    assert 80.0 <= results.mean_angle <= 115.0
    assert np.isfinite(results.std_angle)


# --- Multi-batch test: with split_factor=1 each frame produces its own
# angle, so we should get one angle per frame, not a single collapsed value.
@pytest.mark.integration
def test_binning_contact_angle_analyzer_per_frame_with_split_factor(
    filename, oxygen_indices, binning_params
):
    analyzer = BinningTrajectoryAnalyzer(
        parser=LammpsDumpParser(filename),
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        width_cylinder=21,
        binning_params=binning_params,
    )

    # split_factor=1 → one batch per frame → 3 batch-level angles.
    results = analyzer.analyze([1, 2, 3], split_factor=1)

    assert results.method_metadata == {"frames_per_trajectory": 1}
    assert results.angles_per_batch.shape == (3,)
    # Each batch can either converge to a physically-plausible angle in
    # [0, 180] or return NaN (signaling fit failure on a single frame).
    for angle in results.angles_per_batch:
        assert np.isnan(angle) or (0.0 <= angle <= 180.0)
