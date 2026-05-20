import pathlib

import numpy as np
import pytest
from wetting_angle_kit.contact_angle_methods import contact_angle_analyzer
from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder


# --- Fixtures ---
@pytest.fixture
def filename():
    return (
        pathlib.Path(__file__).parent
        / ".."
        / "trajectories"
        / "traj_spherical_drop_4k.lammpstrj"
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


# --- Unit Tests for ContactAngleSliced ---
@pytest.mark.integration
@pytest.mark.slow
def test_contact_angle_sliced_with_real_data(parser, oxygen_indices):
    # Parse liquid positions for frame 0
    liquid_positions = parser.parse(frame_index=0, indices=oxygen_indices)
    max_dist = int(
        np.max(
            np.array(
                [parser.box_size_y(frame_index=0), parser.box_size_x(frame_index=0)]
            )
        )
        / 2
    )
    mean_liquid_position = np.mean(liquid_positions, axis=0)

    # Initialize ContactAngleSliced
    from wetting_angle_kit.contact_angle_methods.sliced import (
        ContactAngleSliced,
    )

    predictor = ContactAngleSliced(
        liquid_coordinates=liquid_positions,
        liquid_geom_center=mean_liquid_position,
        droplet_geometry="spherical",
        delta_gamma=20,
        max_dist=max_dist,
    )

    # Test predict_contact_angle
    angles, surfaces, popt_arrays = predictor.predict_contact_angle()
    assert isinstance(angles, list)
    assert isinstance(surfaces, list)
    assert isinstance(popt_arrays, list)
    assert len(angles) > 0


# --- Integration Test for SlicedContactAngleAnalyzer ---
@pytest.mark.integration
@pytest.mark.slow
def test_sliced_contact_angle_analyzer_with_real_data(
    filename, oxygen_indices, tmp_path
):
    # Use a temporary directory for output
    output_dir = tmp_path / "result_dump_spherical_sliced"

    analyzer = contact_angle_analyzer(
        method="sliced",
        parser=LammpsDumpParser(filename),
        output_dir=output_dir,
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        delta_gamma=20,
    )

    results = analyzer.analyze([1])

    # Assert results
    assert "mean_angle" in results
    assert "std_angle" in results
    assert "angles" in results
    assert len(results["angles"]) == 1
    # The fixture is a water droplet on a graphene-like substrate, which
    # gives a contact angle around 90-100° (literature: ~93° for graphene).
    # Assert a tight physically-plausible band so regressions in the
    # sliced pipeline are caught.
    assert 80.0 <= results["mean_angle"] <= 110.0
    assert np.isfinite(results["std_angle"])
