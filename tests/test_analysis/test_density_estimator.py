"""Density-estimator strategy: binning vs gaussian.

The coupled-fit analyzers accept either
:meth:`DensityEstimator.binning` (the default — top-hat histogram) or
:meth:`DensityEstimator.gaussian` (3D Gaussian KDE on cell centres).
These tests verify both code paths on the LAMMPS fixtures and check
that the Gaussian variant produces a smoother density field than the
histogram on the same grid.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("ovito")

from wetting_angle_kit.analysis import (  # noqa: E402
    CoupledFit2DAnalyzer,
    CoupledFit3DAnalyzer,
    DensityEstimator,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator  # noqa: E402
from wetting_angle_kit.parsers import (  # noqa: E402
    LammpsDumpParser,
    LammpsDumpWaterFinder,
)

_SPHERE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)
_CYL = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_10_3_330w_nve_4k_reajust.lammpstrj"
)


@pytest.fixture
def oxygen_indices_sphere() -> np.ndarray:
    return LammpsDumpWaterFinder(
        _SPHERE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


@pytest.fixture
def oxygen_indices_cyl() -> np.ndarray:
    return LammpsDumpWaterFinder(
        _CYL, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)


@pytest.mark.integration
def test_coupled_fit_2d_gaussian_estimator_recovers_known_angle(
    oxygen_indices_sphere: np.ndarray,
) -> None:
    """``CoupledFit2DAnalyzer`` with the Gaussian estimator lands at ~95°.

    Same grid spec as the binning variant; only the per-cell density
    estimator changes. The Gaussian smoothing reduces Poisson noise
    relative to the histogram, so this combination is usable on
    per-frame batches where the binning variant has occasionally
    landed in degenerate fits.
    """
    binning_params = {
        "xi_0": 0.0,
        "xi_f": 40.0,
        "bin_width_x": 1.0,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "bin_width_z": 1.0,
    }
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(_SPHERE),
        atom_indices=oxygen_indices_sphere,
        droplet_geometry="spherical",
        binning_params=binning_params,
        density_estimator=DensityEstimator.gaussian(density_sigma=2.5),
        temporal_aggregator=TemporalAggregator(batch_size=1),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert 90.0 < batch.angle < 100.0
    assert 25.0 < batch.model_params["R_eq"] < 50.0


@pytest.mark.integration
def test_coupled_fit_2d_gaussian_estimator_smoother_than_binning(
    oxygen_indices_sphere: np.ndarray,
) -> None:
    """At equal grid spec, Gaussian KDE → smoother density than histogram.

    Quantified as the inter-cell coefficient of variation across the
    occupied bulk region: the Gaussian density's CoV is strictly
    lower than the binning density's CoV on the same fixture and grid.
    """
    binning_params = {
        "xi_0": 0.0,
        "xi_f": 40.0,
        "bin_width_x": 1.0,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "bin_width_z": 1.0,
    }

    def _density(estimator: DensityEstimator) -> np.ndarray:
        analyzer = CoupledFit2DAnalyzer(
            parser=LammpsDumpParser(_SPHERE),
            atom_indices=oxygen_indices_sphere,
            droplet_geometry="spherical",
            binning_params=binning_params,
            density_estimator=estimator,
            temporal_aggregator=TemporalAggregator(batch_size=1),
        )
        return analyzer.analyze([1]).batches[0].density

    rho_bin = _density(DensityEstimator.binning())
    rho_gauss = _density(DensityEstimator.gaussian(density_sigma=2.5))

    # Use the Gaussian density to define the bulk region (it's smooth
    # enough that "above half the bulk" picks out a stable band of
    # cells), then compute the coefficient of variation of both
    # densities on that same set of cells. A smoother density gives a
    # lower CoV across a roughly-uniform bulk region; the binning's
    # Poisson noise should make its CoV strictly larger.
    mask_bulk = rho_gauss > 0.5 * float(rho_gauss.max())
    assert mask_bulk.sum() > 50, "bulk mask too small for the test to mean anything"
    cov_bin = float(np.std(rho_bin[mask_bulk]) / np.mean(rho_bin[mask_bulk]))
    cov_gauss = float(np.std(rho_gauss[mask_bulk]) / np.mean(rho_gauss[mask_bulk]))
    print(
        f"\nbulk CoV (mask = gaussian > 0.5 max): "
        f"binning = {cov_bin:.4f}, gaussian = {cov_gauss:.4f}"
    )
    assert cov_gauss < cov_bin


@pytest.mark.integration
def test_coupled_fit_2d_gaussian_estimator_works_on_cylinder(
    oxygen_indices_cyl: np.ndarray,
) -> None:
    """Gaussian estimator + cylinder geometry lands at ~97-103°.

    Reference: binning estimator on the same fixture recovers ~95-99°
    across frames 0-4 with per-frame std ~3.6°. The Gaussian variant
    shifts the angle ~2-3° higher (the KDE iso-level sits slightly
    inside the geometric edge) and gives similar per-frame std.
    """
    binning_params = {
        "xi_0": 0.0,
        "xi_f": 70.0,
        "bin_width_x": 1.0,
        "zi_0": 0.0,
        "zi_f": 80.0,
        "bin_width_z": 1.0,
    }
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(_CYL),
        atom_indices=oxygen_indices_cyl,
        droplet_geometry="cylinder_y",
        binning_params=binning_params,
        density_estimator=DensityEstimator.gaussian(density_sigma=2.5),
        temporal_aggregator=TemporalAggregator(batch_size=1),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert 95.0 < batch.angle < 110.0


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_fit_3d_gaussian_estimator_recovers_known_angle(
    oxygen_indices_sphere: np.ndarray,
) -> None:
    """``CoupledFit3DAnalyzer`` with the Gaussian estimator lands at ~96°.

    The 3D coupled fit is much more constrained than the 2D per-frame
    case: per-frame std across frames 0-4 is ~0.9° for the Gaussian
    variant and ~0.7° for the binning variant. Mean angles agree
    within ~0.5° (gaussian ~96.5°, binning ~95.6°), so the band
    here is narrow on purpose — a regression in the estimator
    plumbing or volume normalisation would push the angle well
    outside it.
    """
    binning_params = {
        "xi_0": -40.0,
        "xi_f": 40.0,
        "bin_width_x": 3.3,
        "yi_0": -40.0,
        "yi_f": 40.0,
        "bin_width_y": 3.3,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "bin_width_z": 1.6,
    }
    analyzer = CoupledFit3DAnalyzer(
        parser=LammpsDumpParser(_SPHERE),
        atom_indices=oxygen_indices_sphere,
        droplet_geometry="spherical",
        binning_params=binning_params,
        density_estimator=DensityEstimator.gaussian(density_sigma=3.0),
        temporal_aggregator=TemporalAggregator(batch_size=1),
    )
    batch = analyzer.analyze([1]).batches[0]
    assert 93.0 < batch.angle < 100.0


def test_density_estimator_kind_tags_are_distinct() -> None:
    """The two strategies expose distinct ``kind`` tags for the tqdm label."""
    assert DensityEstimator.binning().kind == "binning"
    assert DensityEstimator.gaussian().kind == "gaussian"
