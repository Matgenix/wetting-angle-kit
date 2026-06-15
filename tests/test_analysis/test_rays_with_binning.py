"""``rays`` + binning extractor + parity vs ``rays`` + Gaussian.

Four flavors:

- **HistogramDensityField unit test.** Top-hat evaluator returns the
  uniform bulk density inside a dense atom box and zero well outside.
- **Fibonacci sphere recovery.** ``rays`` + binning + whole+spherical
  on a known sphere; recovered R sits near truth.
- **Slicing-mode parity vs rays + Gaussian.** Same atom cloud, same
  ray fan, comparable kernel widths (top-hat diameter chosen to match
  Gaussian FWHM). Recovered per-slice interface points should agree
  within a small absolute tolerance.
- **End-to-end angle parity on the LAMMPS fixture.** Both ray
  extractors, paired with the slicing fitter and ``min_plus_offset``
  wall, should produce angles within a few degrees of each other.
"""

import pathlib

import numpy as np
import pytest

from wetting_angle_kit.analysis._density import (
    HistogramDensityField,
    fit_tanh_profiles_batched,
)
from wetting_angle_kit.analysis.density_estimator import DensityEstimator
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface import InterfaceExtractor, SpaceSampling


def _uniform_sphere_atoms(radius: float, n_atoms: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while sum(p.shape[0] for p in pts) < n_atoms:
        sample = rng.uniform(-radius, radius, size=(4 * n_atoms, 3))
        pts.append(sample[np.linalg.norm(sample, axis=1) < radius])
    return np.concatenate(pts, axis=0)[:n_atoms]


def test_fit_tanh_profiles_batched_resolves_clean_rays_and_nans_flat() -> None:
    """The batched LM fit recovers the interface of resolvable rays and
    reports ``nan`` for a flat ray that carries no interface."""
    z = np.linspace(0.0, 30.0, 60)
    # ``rho = d * tanh(zd - s) + h``: liquid (high) near s=0, vapour
    # (low) past the interface. Two clean rays at zd=15 and zd=10.
    clean_15 = 0.03 * np.tanh(-(z - 15.0)) + 0.03
    clean_10 = 0.04 * np.tanh(-(z - 10.0)) + 0.04
    flat = np.full_like(z, 0.05)
    out = fit_tanh_profiles_batched(z, np.stack([clean_15, flat, clean_10]))
    assert out[0] == pytest.approx(15.0, abs=0.2)
    assert np.isnan(out[1])  # flat profile -> no resolvable interface
    assert out[2] == pytest.approx(10.0, abs=0.2)


def test_fit_tanh_profiles_batched_recovers_noisy_ray() -> None:
    """A resolvable interface is still recovered under moderate noise,
    and the result stays clipped within the sampling envelope."""
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 40.0, 120)
    rho = 0.033 * np.tanh(-(z - 22.0)) + 0.033
    noisy = rho + rng.normal(scale=2e-3, size=z.shape)
    out = fit_tanh_profiles_batched(z, noisy[None, :])
    assert out[0] == pytest.approx(22.0, abs=1.0)
    assert 0.0 <= out[0] <= float(z[-1])


def test_histogram_density_field_uniform_box() -> None:
    """Bulk density check on a dense uniform-volume box of atoms.

    Single-sample Poisson noise is large at small bin radii; we
    instead probe a grid of N interior points and check the **mean**
    density across that grid lands near the bulk value. Far outside
    the box the field returns zero.
    """
    rng = np.random.default_rng(0)
    half = 30.0
    n_atoms = 50000
    atoms = rng.uniform(-half, half, size=(n_atoms, 3))
    bulk_density = n_atoms / (2 * half) ** 3  # atoms / Å³

    field = HistogramDensityField(atoms, bin_width=6.0)
    # 50 interior sample points, comfortably away from the box wall.
    grid_lo, grid_hi = -half + 5.0, half - 5.0
    inside = rng.uniform(grid_lo, grid_hi, size=(50, 3))
    dens_in = field.evaluate(inside)
    # Outside: far from atoms.
    outside = np.array([[0.0, 0.0, 100.0], [50.0, 50.0, 50.0]])
    dens_out = field.evaluate(outside)

    mean_in = float(np.mean(dens_in))
    rel_dev_mean = abs(mean_in - bulk_density) / bulk_density
    print(
        f"\nHistogramDensityField bulk = {bulk_density:.5f} atoms/Å³, "
        f"inside mean over 50 samples = {mean_in:.5f}, "
        f"|rel|_mean = {rel_dev_mean:.3f}, "
        f"outside = {dens_out.tolist()}"
    )
    # Averaging cancels Poisson noise: the mean should land within 5%
    # of the true bulk density.
    assert rel_dev_mean < 0.05
    assert np.all(dens_out == 0.0)


def test_rays_with_binning_whole_spherical_recovers_sphere() -> None:
    """rays + binning + whole+spherical on a known sphere → R near truth.

    Top-hat radius ``bin_width / 2`` defines the per-sample
    neighbourhood; pick it big enough that each bin captures O(50)
    atoms so the tanh fit is fed a smooth profile rather than
    Poisson-noisy bins.
    """
    radius = 20.0
    bin_width = 8.0  # ~270 Å³ bin volume × 0.45 atoms/Å³ ≈ 120 atoms / bin
    atoms = _uniform_sphere_atoms(radius=radius, n_atoms=15000, seed=0)

    extractor = InterfaceExtractor(
        sampling=SpaceSampling.rays(n_rays_sphere=400, points_per_angstrom=2.0),
        density=DensityEstimator.binning(bin_width=bin_width),
    )
    geom = DropletGeometry.coerce("spherical")
    shell = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="whole",
    )
    assert isinstance(shell, np.ndarray)
    assert shell.shape == (400, 3)
    r = np.linalg.norm(shell, axis=1)
    print(
        f"\nrays + binning sphere recovery: R_truth = {radius} Å, "
        f"R_mean = {float(np.mean(r)):.3f} Å, R_std = {float(np.std(r)):.3f} Å"
    )
    # The half-density point sits ≤ d/2 = 4 Å from the geometric edge
    # — i.e. the recovered R is inward-shifted by at most that much.
    assert abs(float(np.mean(r)) - radius) < bin_width / 2.0
    assert float(np.std(r)) < 1.5


def test_rays_with_binning_matches_rays_with_gaussian_slicing_spherical() -> None:
    """rays + binning ≈ rays + Gaussian within a small tolerance on a slicing fixture.

    Top-hat diameter is chosen by variance match
    (``d ≈ σ √12``) so the two kernels produce comparable smoothing.
    The dense droplet (15 000 atoms) and 2 samples / Å along each ray
    keep the binned profiles smooth enough for the tanh fit.
    """
    atoms = _uniform_sphere_atoms(radius=20.0, n_atoms=15000, seed=2)
    sigma = 3.0
    bin_width = sigma * float(np.sqrt(12.0))  # variance-matched top-hat
    delta_azimuthal = 30.0
    delta_polar = 8.0

    geom = DropletGeometry.coerce("spherical")
    g = InterfaceExtractor(
        sampling=SpaceSampling.rays(
            delta_azimuthal=delta_azimuthal,
            delta_polar=delta_polar,
            points_per_angstrom=2.0,
        ),
        density=DensityEstimator.gaussian(density_sigma=sigma),
    )
    b = InterfaceExtractor(
        sampling=SpaceSampling.rays(
            delta_azimuthal=delta_azimuthal,
            delta_polar=delta_polar,
            points_per_angstrom=2.0,
        ),
        density=DensityEstimator.binning(bin_width=bin_width),
    )
    gauss_slices = g.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )
    bin_slices = b.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        surface_kind="slicing",
    )
    assert isinstance(gauss_slices, list)
    assert isinstance(bin_slices, list)
    assert len(gauss_slices) == len(bin_slices)

    max_diff = 0.0
    mean_diff = 0.0
    n_slices = 0
    for gs, bs in zip(gauss_slices, bin_slices, strict=True):
        assert gs.shape == bs.shape
        diff = float(np.max(np.abs(gs - bs)))
        mean_diff += float(np.mean(np.abs(gs - bs)))
        max_diff = max(max_diff, diff)
        n_slices += 1
    mean_diff /= n_slices
    print(
        f"\nslicing+spherical, σ={sigma}, d=σ√12={bin_width:.3f}: "
        f"max |gauss - binning| = {max_diff:.3f} Å, "
        f"mean = {mean_diff:.3f} Å (over {n_slices} slices)"
    )
    # Comparable smoothing should leave per-slice points within ~σ
    # on average; max can spike to a few σ near rays where the
    # binned profile is noisier.
    assert mean_diff < sigma
    assert max_diff < 4.0 * sigma


_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)


@pytest.mark.integration
@pytest.mark.slow
def test_rays_with_binning_end_to_end_angle_close_to_rays_with_gaussian() -> None:
    """Both ray extractors → similar angle on the LAMMPS water/graphene fixture."""
    pytest.importorskip("ovito")
    from wetting_angle_kit.analysis import (
        DensityEstimator,
        SurfaceFitter,
        TrajectoryAnalyzer,
        WallDetector,
    )
    from wetting_angle_kit.parsers import (
        LammpsDumpParser,
        LammpsDumpWaterFinder,
    )

    finder = LammpsDumpWaterFinder(_FIXTURE, oxygen_type=1, hydrogen_type=2)
    oxygen_indices = finder.get_water_oxygen_indices(0)

    def _angle(extractor: InterfaceExtractor) -> float:
        analyzer = TrajectoryAnalyzer(
            parser=LammpsDumpParser(_FIXTURE),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
            interface_extractor=extractor,
            surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
            wall_detector=WallDetector.min_plus_offset(offset=0.0),
        )
        results = analyzer.analyze([1])
        return float(results.per_batch_angles[0])

    sigma = 3.0
    bin_width = sigma * float(np.sqrt(12.0))  # variance-matched top-hat
    angle_g = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.rays(
                delta_azimuthal=20.0, delta_polar=8.0, points_per_angstrom=2.0
            ),
            density=DensityEstimator.gaussian(density_sigma=sigma),
        )
    )
    angle_b = _angle(
        InterfaceExtractor(
            sampling=SpaceSampling.rays(
                delta_azimuthal=20.0, delta_polar=8.0, points_per_angstrom=2.0
            ),
            density=DensityEstimator.binning(bin_width=bin_width),
        )
    )
    drift = abs(angle_g - angle_b)
    print(
        f"\nrays + Gaussian (σ={sigma})    angle = {angle_g:.3f}°"
        f"\nrays + binning (d={bin_width:.3f}) angle = {angle_b:.3f}°"
        f"\n|drift|                     = {drift:.3f}°"
    )
    # Both fall in the same physically-plausible band, drift is a few °.
    assert 70.0 < angle_g < 110.0
    assert 70.0 < angle_b < 110.0
    assert drift < 5.0
