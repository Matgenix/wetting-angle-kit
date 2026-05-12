import pathlib

import numpy as np
import pytest

from wetting_angle_kit.contact_angle_method import contact_angle_analyzer
from wetting_angle_kit.parser import DumpParser, DumpWaterMoleculeFinder


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
    return DumpWaterMoleculeFinder(
        filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2
    )


@pytest.fixture
def oxygen_indices(wat_find):
    return wat_find.get_water_oxygen_ids(0)


@pytest.fixture
def parser(filename):
    return DumpParser(filename)


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


# --- Unit Test for BinningContactAngleAnalyzer ---
@pytest.mark.integration
def test_binning_contact_angle_analyzer_with_real_data(
    filename, oxygen_indices, binning_params, tmp_path
):
    # Use a temporary directory for output
    output_dir = tmp_path / "result_dump_cylinder_noplot"

    # Create the analyzer
    analyzer = contact_angle_analyzer(
        method="binning",
        parser=DumpParser(filename),
        output_dir=output_dir,
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        width_cylinder=21,
        binning_params=binning_params,
        plot_graphs=False,
    )

    # Run analysis for frame 1
    results = analyzer.analyze([1])

    # Assert results
    assert "mean_angle" in results
    assert "std_angle" in results
    assert "angles" in results
    assert len(results["angles"]) == 1
    # Cylindrical droplet on a graphene-like surface gives a contact angle
    # around 90-100° here. Use a moderate band so the test catches gross
    # regressions but tolerates the inherent noise of a single-frame fit.
    assert 80.0 <= results["mean_angle"] <= 115.0
    assert np.isfinite(results["std_angle"])


# --- Multi-batch test: with split_factor=1 each frame produces its own
# angle, so we should get one angle per frame, not a single collapsed value.
@pytest.mark.integration
def test_binning_contact_angle_analyzer_per_frame_with_split_factor(
    filename, oxygen_indices, binning_params, tmp_path
):
    output_dir = tmp_path / "result_dump_per_frame"

    analyzer = contact_angle_analyzer(
        method="binning",
        parser=DumpParser(filename),
        output_dir=output_dir,
        atom_indices=oxygen_indices,
        droplet_geometry="cylinder_y",
        width_cylinder=21,
        binning_params=binning_params,
        plot_graphs=False,
    )

    # split_factor=1 → one batch per frame → 3 batch-level angles.
    results = analyzer.analyze([1, 2, 3], split_factor=1)

    assert results["method_metadata"] == {"frames_per_trajectory": 1}
    assert results["angles"].shape == (3,)
    # Each batch can either converge to a physically-plausible angle in
    # [0, 180] or return NaN (signaling fit failure on a single frame).
    for angle in results["angles"]:
        assert np.isnan(angle) or (0.0 <= angle <= 180.0)
