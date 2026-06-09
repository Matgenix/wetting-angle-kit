"""Phase 7 quantification: grid extractors (slicing-only).

Three flavors:

- **Synthetic spherical droplet → recovered angle ≈ truth.** Generates
  atoms inside a spherical cap of known truth angle, runs the
  grid extractor + slicing fitter pipeline, and checks the recovered
  angle.
- **grid_binning vs grid_gaussian on the same atoms.** Both should
  recover a similar interface; the Gaussian-smoothed variant is
  cleaner (lower per-slice RMS) than the bare histogram.
- **End-to-end on the LAMMPS water/graphene fixture.** Grid extractors
  paired with the slicing fitter should produce angles within a few
  degrees of ``rays_gaussian`` on the same fixture/frame.
"""

import pathlib

import numpy as np
import pytest

# Skip Phase 7 entirely if the optional scikit-image extra isn't
# installed — both grid extractors depend on it.
pytest.importorskip("skimage")

from wetting_angle_kit.analysis import (  # noqa: E402
    InterfaceExtractor,
    SurfaceFitter,
    WallDetector,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry  # noqa: E402


def _spherical_cap_atoms(
    *,
    R: float,
    zc: float,
    z_wall: float,
    n_atoms: int,
    seed: int = 0,
) -> np.ndarray:
    """Atoms uniformly filling a spherical cap.

    Sphere of radius ``R`` centered at ``(0, 0, zc)``; only atoms with
    ``z > z_wall`` are kept (the visible cap above the wall).
    """
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while sum(p.shape[0] for p in pts) < n_atoms:
        sample = rng.uniform(-R, R, size=(4 * n_atoms, 3))
        inside_sphere = np.linalg.norm(sample, axis=1) < R
        sample = sample[inside_sphere]
        sample[:, 2] += zc
        sample = sample[sample[:, 2] > z_wall]
        pts.append(sample)
    return np.concatenate(pts, axis=0)[:n_atoms]


def _default_grid_params(max_dist: float) -> dict[str, object]:
    return {
        "xi_0": 0.0,
        "xi_f": max_dist,
        "nbins_xi": 50,
        "zi_0": 0.0,
        "zi_f": max_dist,
        "nbins_zi": 50,
    }


def test_grid_gaussian_recovers_known_spherical_cap_angle() -> None:
    """Synthetic spherical cap → ``grid_gaussian`` recovers truth within ~1°."""
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=15000, seed=0)

    extractor = InterfaceExtractor.grid_gaussian(
        grid_params=_default_grid_params(max_dist=35.0),
        density_sigma=2.0,
    )
    geom = DropletGeometry.coerce("spherical")
    extractor.validate_compatibility(surface_kind="slicing", droplet_geometry=geom)
    contours = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=35.0,
        surface_kind="slicing",
    )
    assert isinstance(contours, list)
    assert len(contours) == 1
    contour = contours[0]
    assert contour.ndim == 2 and contour.shape[1] == 2
    assert len(contour) >= 10

    # ``find_contours`` traces the iso-line all the way around the
    # droplet — including a thin "floor" segment near ``z = z_wall``
    # that arises from histogram discretisation across the wall.
    # ``SurfaceFitter.slicing.surface_filter_offset`` is the designed
    # mechanism for dropping that floor before the Kasa circle fit.
    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=contours, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid_gaussian cap recovery: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, |drift| = {drift:.3f}°, "
        f"R_fit = {out.slice_popts[0][2]:.3f}, "
        f"zc_fit = {out.slice_popts[0][1]:.3f}, "
        f"contour_points = {len(contour)}, "
        f"rms_residual = {out.rms_residual:.3f} Å"
    )
    # Grid resolution 0.71 Å + Gaussian smoothing σ=2 ⇒ a fraction of
    # a degree on this dense fixture.
    assert drift < 2.0


def test_grid_binning_recovers_known_spherical_cap_with_coarse_bins() -> None:
    """``grid_binning`` (no smoothing) needs coarser bins to give a usable contour.

    On finer grids the Poisson noise per bin dominates and the
    iso-line jitter spoils the Kasa fit; the coarse-bins case shows
    the tool produces sensible answers with the right configuration.
    """
    R, zc, z_wall = 25.0, 0.0, 5.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=50000, seed=0)

    # 24 cells span the 35-Å envelope → dxi ≈ 1.5 Å, giving ~10×
    # more atoms per bin than the nbins=50 default at this density.
    grid_params: dict[str, object] = {
        "xi_0": 0.0,
        "xi_f": 35.0,
        "nbins_xi": 25,
        "zi_0": 0.0,
        "zi_f": 35.0,
        "nbins_zi": 25,
    }
    extractor = InterfaceExtractor.grid_binning(grid_params=grid_params)
    geom = DropletGeometry.coerce("spherical")
    contours = extractor.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=35.0,
        surface_kind="slicing",
    )

    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out = fitter.fit(interface_data=contours, z_wall=z_wall, droplet_geometry=geom)
    drift = abs(out.angle - truth_angle)
    print(
        f"\ngrid_binning (coarse) cap recovery: truth = {truth_angle:.3f}°, "
        f"recovered = {out.angle:.3f}°, |drift| = {drift:.3f}°, "
        f"R_fit = {out.slice_popts[0][2]:.3f}, "
        f"rms_residual = {out.rms_residual:.3f} Å"
    )
    assert drift < 5.0


def test_grid_gaussian_smoother_than_grid_binning() -> None:
    """At equal grid spec, ``grid_gaussian`` gives a smoother contour
    than ``grid_binning``.

    Same atoms, same grid, equal-shape contours; the smoothed variant
    should have a lower per-point Kasa-fit residual.
    """
    R, zc, z_wall = 25.0, 0.0, 5.0
    atoms = _spherical_cap_atoms(R=R, zc=zc, z_wall=z_wall, n_atoms=50000, seed=1)

    # Coarse-enough grid for both estimators to give usable contours.
    grid_params: dict[str, object] = {
        "xi_0": 0.0,
        "xi_f": 35.0,
        "nbins_xi": 25,
        "zi_0": 0.0,
        "zi_f": 35.0,
        "nbins_zi": 25,
    }
    geom = DropletGeometry.coerce("spherical")

    b = InterfaceExtractor.grid_binning(grid_params=grid_params)
    g = InterfaceExtractor.grid_gaussian(grid_params=grid_params, density_sigma=2.0)

    binning_contours = b.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=35.0,
        surface_kind="slicing",
    )
    gaussian_contours = g.extract(
        liquid_coordinates=atoms,
        center_geom=np.zeros(3),
        droplet_geometry=geom,
        max_dist=35.0,
        surface_kind="slicing",
    )

    fitter = SurfaceFitter.slicing(surface_filter_offset=3.0)
    out_b = fitter.fit(
        interface_data=binning_contours, z_wall=z_wall, droplet_geometry=geom
    )
    out_g = fitter.fit(
        interface_data=gaussian_contours, z_wall=z_wall, droplet_geometry=geom
    )
    print(
        f"\ngrid_binning  rms = {out_b.rms_residual:.3f} Å, angle = {out_b.angle:.3f}°"
        f"\ngrid_gaussian rms = {out_g.rms_residual:.3f} Å, angle = {out_g.angle:.3f}°"
    )
    assert out_g.rms_residual <= out_b.rms_residual
    truth_angle = float(np.degrees(np.arccos((z_wall - zc) / R)))
    assert abs(out_b.angle - truth_angle) < 8.0
    assert abs(out_g.angle - truth_angle) < 5.0


@pytest.mark.integration
@pytest.mark.slow
def test_grid_extractors_end_to_end_close_to_rays_gaussian() -> None:
    """Grid extractor angles on the LAMMPS fixture sit within a few ° of rays."""
    pytest.importorskip("ovito")
    from wetting_angle_kit.analysis import TrajectoryAnalyzer
    from wetting_angle_kit.parsers import (
        LammpsDumpParser,
        LammpsDumpWaterFinder,
    )

    fixture = (
        pathlib.Path(__file__).parent
        / ".."
        / "trajectories"
        / "traj_spherical_drop_4k.lammpstrj"
    )
    finder = LammpsDumpWaterFinder(fixture, oxygen_type=1, hydrogen_type=2)
    oxygen_indices = finder.get_water_oxygen_ids(0)

    # ``grid_gaussian`` works on the 4k-oxygen fixture with the same
    # grid as the synthetic test. ``grid_binning`` (no smoothing) needs
    # coarser bins to avoid noisy iso-contour points; we evaluate each
    # at its own well-suited bin count.
    grid_params_gauss = {
        "xi_0": 0.0,
        "xi_f": 40.0,
        "nbins_xi": 26,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "nbins_zi": 26,
    }
    grid_params_bin = {
        "xi_0": 0.0,
        "xi_f": 40.0,
        "nbins_xi": 16,
        "zi_0": 0.0,
        "zi_f": 40.0,
        "nbins_zi": 16,
    }

    def _angle(extractor: InterfaceExtractor) -> float:
        analyzer = TrajectoryAnalyzer(
            parser=LammpsDumpParser(fixture),
            atom_indices=oxygen_indices,
            droplet_geometry="spherical",
            interface_extractor=extractor,
            surface_fitter=SurfaceFitter.slicing(surface_filter_offset=3.0),
            wall_detector=WallDetector.min_plus_offset(offset=0.0),
        )
        return float(analyzer.analyze([1]).per_batch_angles[0])

    angle_rays = _angle(
        InterfaceExtractor.rays_gaussian(
            delta_azimuthal=20.0, delta_polar=8.0, density_sigma=3.0
        )
    )
    angle_grid_bin = _angle(
        InterfaceExtractor.grid_binning(grid_params=grid_params_bin)
    )
    angle_grid_gauss = _angle(
        InterfaceExtractor.grid_gaussian(
            grid_params=grid_params_gauss, density_sigma=2.0
        )
    )
    print(
        f"\nrays_gaussian          angle = {angle_rays:.3f}°"
        f"\ngrid_binning (nbins=16) angle = {angle_grid_bin:.3f}°  "
        f"|drift| = {abs(angle_grid_bin - angle_rays):.3f}°"
        f"\ngrid_gaussian (nbins=26) angle = {angle_grid_gauss:.3f}°  "
        f"|drift| = {abs(angle_grid_gauss - angle_rays):.3f}°"
    )
    for angle in (angle_rays, angle_grid_bin, angle_grid_gauss):
        assert 70.0 < angle < 110.0
    # ``grid_gaussian`` smooths the density → close to ``rays_gaussian``.
    assert abs(angle_grid_gauss - angle_rays) < 5.0
    # ``grid_binning`` is intrinsically noisier; allow a wider band.
    assert abs(angle_grid_bin - angle_rays) < 12.0
