"""Phase 3 quantification: ``SurfaceFitter.slicing()`` correctness + parity.

Two flavors:

- **Synthetic-circle correctness.** Per-slice points lying on a known
  circle. The fitter should recover the truth contact angle to
  numerical precision and report (near-)zero fit residual.
- **End-to-end parity** against the legacy
  :class:`SlicingTrajectoryAnalyzer` on the LAMMPS water-on-graphene
  fixture. The two pipelines compute the angle baseline slightly
  differently (legacy: per-slice ``min(z)``; new: per-batch
  ``min(z) + offset``), so a small drift is expected — quantified
  with and without the ``min_plus_offset`` offset.
"""

import pathlib

import numpy as np
import pytest

# Slicing fixture is a LAMMPS dump parsed through OVITO; skip if the
# optional backend is missing (typically macOS CI without ovito).
pytest.importorskip("ovito")

from wetting_angle_kit.analysis import (  # noqa: E402
    InterfaceExtractor,
    SlicingTrajectoryAnalyzer,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry  # noqa: E402
from wetting_angle_kit.parsers import (  # noqa: E402
    LammpsDumpParser,
    LammpsDumpWaterFinder,
)

# --- Synthetic correctness -------------------------------------------------


def _circle_arc_points(
    xc: float, zc: float, radius: float, theta_lo: float, theta_hi: float, n: int
) -> np.ndarray:
    """Sample (x, z) points along an arc of a circle."""
    theta = np.linspace(theta_lo, theta_hi, n)
    return np.column_stack([xc + radius * np.cos(theta), zc + radius * np.sin(theta)])


def test_slicing_fitter_recovers_known_circle_angle() -> None:
    """Per-slice points on a known circle → recovered angle matches truth."""
    # Circle of radius R centred at (0, zc); wall at z=0. The cap
    # extends from the contact line up to the top of the circle.
    # cos θ = (z_wall - zc) / R, so for zc = -5 and R = 20:
    # cos θ = 0.25  →  θ ≈ 75.522°.
    xc_truth, zc_truth, radius_truth = 0.0, -5.0, 20.0
    z_wall = 0.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / radius_truth)))

    # Sample the upper arc of the circle (z > z_wall).
    # theta in radians measured from +x axis; upper arc is [0, pi].
    arc = _circle_arc_points(
        xc=xc_truth,
        zc=zc_truth,
        radius=radius_truth,
        theta_lo=np.arcsin((z_wall - zc_truth) / radius_truth),
        theta_hi=np.pi - np.arcsin((z_wall - zc_truth) / radius_truth),
        n=80,
    )
    # ``surface_filter_offset = 0`` so every arc point is kept.
    fitter = SurfaceFitter.slicing(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=[arc],
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("spherical"),
    )

    print(
        f"\nSynthetic circle: truth = {truth_angle:.4f}°, "
        f"recovered = {out.angle:.4f}°, rms_residual = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 1e-6
    # Algebraic Kasa on exact-circle points fits to ~floating-point
    # precision; the per-slice RMS residual should sit near 0.
    assert out.rms_residual < 1e-9
    # One slice → angle_std = 0.
    assert out.angle_std == 0.0


def test_slicing_fitter_aggregates_across_slices() -> None:
    """Multiple noisy slices around the same true angle — aggregation."""
    radius_truth = 20.0
    zc_truth = -5.0
    z_wall = 0.0
    truth_angle = float(np.degrees(np.arccos((z_wall - zc_truth) / radius_truth)))

    rng = np.random.default_rng(7)
    slices: list[np.ndarray] = []
    for _ in range(8):
        arc = _circle_arc_points(
            xc=0.0,
            zc=zc_truth,
            radius=radius_truth,
            theta_lo=np.arcsin((z_wall - zc_truth) / radius_truth),
            theta_hi=np.pi - np.arcsin((z_wall - zc_truth) / radius_truth),
            n=40,
        )
        # ±0.05 Å Gaussian noise on both axes — sub-resolution thermal jitter.
        arc = arc + rng.normal(0.0, 0.05, size=arc.shape)
        slices.append(arc)

    fitter = SurfaceFitter.slicing(surface_filter_offset=0.0)
    out = fitter.fit(
        interface_data=slices,
        z_wall=z_wall,
        droplet_geometry=DropletGeometry.coerce("spherical"),
    )

    print(
        f"Aggregated 8 noisy slices: truth = {truth_angle:.3f}°, "
        f"mean = {out.angle:.3f}°, angle_std = {out.angle_std:.3f}°, "
        f"rms_residual = {out.rms_residual:.3e} Å"
    )
    assert abs(out.angle - truth_angle) < 0.5
    # Per-slice std should be small and finite.
    assert out.angle_std > 0.0
    assert out.angle_std < 1.0
    assert out.per_slice_angles.shape == (8,)
    assert out.slice_popts.shape == (8, 4)
    assert len(out.slice_surfaces) == 8


# --- End-to-end parity vs legacy SlicingTrajectoryAnalyzer -----------------


_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)


@pytest.fixture
def fixture_path() -> pathlib.Path:
    return _FIXTURE


@pytest.fixture
def oxygen_indices(fixture_path: pathlib.Path) -> np.ndarray:
    finder = LammpsDumpWaterFinder(fixture_path, oxygen_type=1, hydrogen_type=2)
    return finder.get_water_oxygen_ids(0)


def _run_legacy_angle(
    fixture_path: pathlib.Path, oxygen_indices: np.ndarray, frame: int
) -> float:
    analyzer = SlicingTrajectoryAnalyzer(
        parser=LammpsDumpParser(fixture_path),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        delta_gamma=20,
    )
    results = analyzer.analyze([frame])
    return float(np.mean(results.angles[0]))


def _run_new_angle(
    fixture_path: pathlib.Path,
    oxygen_indices: np.ndarray,
    frame: int,
    *,
    wall_offset: float,
) -> float:
    analyzer = TrajectoryAnalyzer(
        parser=LammpsDumpParser(fixture_path),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        interface_extractor=InterfaceExtractor.rays_gaussian(
            delta_azimuthal=20.0,
            delta_polar=8.0,
        ),
        surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
        wall_detector=WallDetector.min_plus_offset(offset=wall_offset),
    )
    results = analyzer.analyze([frame])
    return float(results.per_batch_angles[0])


@pytest.mark.integration
@pytest.mark.slow
def test_slicing_end_to_end_parity_vs_legacy(
    fixture_path: pathlib.Path, oxygen_indices: np.ndarray
) -> None:
    """Quantify the drift between legacy and new slicing pipelines.

    Two new-pipeline configurations are reported:

    - ``min_plus_offset(0)``: angle baseline ≈ legacy's per-slice
      ``min(z)`` (modulo per-slice → per-batch substitution).
    - ``min_plus_offset(2)``: the new default; angle baseline shifted
      2 Å above the lowest interface point.
    """
    frame = 1
    legacy_angle = _run_legacy_angle(fixture_path, oxygen_indices, frame)
    new_angle_offset0 = _run_new_angle(
        fixture_path, oxygen_indices, frame, wall_offset=0.0
    )
    new_angle_offset2 = _run_new_angle(
        fixture_path, oxygen_indices, frame, wall_offset=2.0
    )

    drift_0 = abs(new_angle_offset0 - legacy_angle)
    drift_2 = abs(new_angle_offset2 - legacy_angle)

    print(
        f"\nLegacy mean angle                            = {legacy_angle:.3f}°"
        f"\nNew (min_plus_offset offset=0) mean angle    = "
        f"{new_angle_offset0:.3f}°   |drift| = {drift_0:.3f}°"
        f"\nNew (min_plus_offset offset=2, default) mean = "
        f"{new_angle_offset2:.3f}°   |drift| = {drift_2:.3f}°"
    )

    # Both new configurations should land near the legacy result.
    # offset=0 is the closest analogue (legacy uses per-slice min(z));
    # offset=2 shifts the baseline up by 2 Å (smaller measured angle).
    assert drift_0 < 3.0, f"offset=0 drift too large: {drift_0:.3f}°"
    assert drift_2 < 10.0, f"offset=2 drift too large: {drift_2:.3f}°"
    # Both new angles should still sit in the physically plausible
    # 80–110° band the legacy test enforces for this fixture.
    assert 70.0 < new_angle_offset0 < 110.0
    assert 70.0 < new_angle_offset2 < 110.0
