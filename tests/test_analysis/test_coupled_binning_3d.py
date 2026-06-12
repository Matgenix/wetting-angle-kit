"""Phase 10 quantification: ``CoupledBinning3DAnalyzer``.

Three flavors:

- **3D model on a clean analytic density grid.** Build the analytic
  9-parameter tanh field on a grid, run the model fit, verify the
  recovered contact angle matches truth to ≤ 0.1°.
- **Cylinder rejection.** The analyzer refuses cylindrical droplets
  at construction with the documented pointer to the 2D variant.
- **End-to-end vs ``CoupledBinning2DAnalyzer`` on the LAMMPS fixture.**
  The 2D analyzer collapses the droplet via radial symmetry; the 3D
  one keeps the full 3D density. For an approximately axisymmetric
  droplet the two should agree within a few degrees.
"""

import pathlib

import numpy as np
import pytest

from wetting_angle_kit.analysis.coupled_binning.analyzer_3d import (  # noqa: E402
    CoupledBinning3DAnalyzer,
    _HyperbolicTangentModel3D,
)


def test_3d_tanh_model_recovers_known_cap_angle_on_clean_grid() -> None:
    """Analytic 9-parameter density → recovered angle ≤ 0.1° from truth."""
    # Truth parameters for the field.
    rho1_truth = 3.3e-2
    rho2_truth = 1e-3
    R_eq_truth = 25.0
    xi_c_truth = 0.0
    yi_c_truth = 0.0
    zi_c_truth = 0.0
    zi_0_truth = 5.0  # wall plane
    t1_truth = 1.0
    t2_truth = 1.0
    truth_angle = float(np.degrees(np.arccos((zi_0_truth - zi_c_truth) / R_eq_truth)))

    # Build the analytic density on a centred grid (xi, yi span the
    # droplet; zi spans the wall + apex).
    xi_cc = np.linspace(-30.0, 30.0, 25)
    yi_cc = np.linspace(-30.0, 30.0, 25)
    zi_cc = np.linspace(0.0, 35.0, 25)
    XI, YI, ZI = np.meshgrid(xi_cc, yi_cc, zi_cc, indexing="ij")
    r = np.sqrt(
        (XI - xi_c_truth) ** 2 + (YI - yi_c_truth) ** 2 + (ZI - zi_c_truth) ** 2
    )
    g_r = 0.5 * (
        (rho1_truth + rho2_truth)
        - (rho1_truth - rho2_truth) * np.tanh(2 * (r - R_eq_truth) / t1_truth)
    )
    h_z = 0.5 * (1.0 + np.tanh(2 * (ZI - zi_0_truth) / t2_truth))
    density = g_r * h_z

    model = _HyperbolicTangentModel3D()
    model.fit((XI.ravel(), YI.ravel(), ZI.ravel()), density.ravel())
    recovered_angle = model.compute_contact_angle()

    drift = abs(recovered_angle - truth_angle)
    print(
        f"\n3D-tanh analytic recovery: truth = {truth_angle:.4f}°, "
        f"recovered = {recovered_angle:.4f}°, |drift| = {drift:.3e}°, "
        f"R_eq = {model.params[2]:.3f} (truth {R_eq_truth}), "
        f"zi_c = {model.params[5]:.3f} (truth {zi_c_truth}), "
        f"zi_0 = {model.params[6]:.3f} (truth {zi_0_truth})"
    )
    assert drift < 0.1


def test_coupled_binning_3d_rejects_cylinder() -> None:
    """Constructing the analyzer with a cylinder droplet raises clearly."""

    class _MockParser:
        filepath = "_mock_"

        def box_size_x(self, frame_index: int) -> float:
            return 100.0

        def box_size_y(self, frame_index: int) -> float:
            return 100.0

        def frame_count(self) -> int:
            return 1

    # Avoid the detect_parser_type call in the shared base by faking
    # the filepath check. The base validates via filepath only here,
    # so a temporary file would work but the parser-shape error is
    # what would fire first if we passed a real one. The cylinder
    # rejection sits **after** super().__init__, so we expect a
    # ValueError from the cylinder check.
    with pytest.raises(ValueError):
        CoupledBinning3DAnalyzer(
            parser=_MockParser(),
            droplet_geometry="cylinder_y",
            binning_params={
                "xi_0": -30,
                "xi_f": 30,
                "bin_width_x": 6.0,
                "yi_0": -30,
                "yi_f": 30,
                "bin_width_y": 6.0,
                "zi_0": 0,
                "zi_f": 30,
                "bin_width_z": 3.0,
            },
        )


_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_binning_3d_close_to_2d_on_lammps_fixture() -> None:
    """On an axisymmetric droplet the 3D and 2D fits should agree within ~few°."""
    pytest.importorskip("ovito")
    from wetting_angle_kit.analysis import CoupledBinning2DAnalyzer
    from wetting_angle_kit.parsers import (
        LammpsDumpParser,
        LammpsDumpWaterFinder,
    )

    finder = LammpsDumpWaterFinder(_FIXTURE, oxygen_type=1, hydrogen_type=2)
    oxygen_indices = finder.get_water_oxygen_ids(0)

    # 2D analyzer — radial (xi, zi).
    binning_params_2d = {
        "xi_0": 0,
        "xi_f": 40,
        "bin_width_x": 1.0,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "bin_width_z": 1.0,
    }
    legacy_2d = CoupledBinning2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params_2d,
    )
    angle_2d = float(legacy_2d.analyze([1]).batches[0].angle)

    # 3D analyzer — full (xi, yi, zi).
    binning_params_3d = {
        "xi_0": -40,
        "xi_f": 40,
        "bin_width_x": 3.3,
        "yi_0": -40,
        "yi_f": 40,
        "bin_width_y": 3.3,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "bin_width_z": 1.6,
    }
    new_3d = CoupledBinning3DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params=binning_params_3d,
    )
    angle_3d = float(new_3d.analyze([1]).batches[0].angle)

    drift = abs(angle_2d - angle_3d)
    print(
        f"\nCoupledBinning2DAnalyzer angle = {angle_2d:.3f}°"
        f"\nCoupledBinning3DAnalyzer angle = {angle_3d:.3f}°"
        f"\n|drift|                        = {drift:.3f}°"
    )
    # Both should land in the physically plausible band.
    for angle in (angle_2d, angle_3d):
        assert 70.0 < angle < 110.0
    # For an axisymmetric droplet the 3D fit's extra degrees of
    # freedom (xi_c, yi_c) collapse to ~0 and the radial profile
    # mirrors the 2D one; allow up to 8° drift to absorb noise from
    # the sparser 3D grid on the 4k-atom fixture.
    assert drift < 8.0
