"""Coupled-fit contact-angle example.

Runs the coupled hyperbolic-tangent fit on a 2D density grid via
:class:`CoupledFit2DAnalyzer`. The analyzer solves interface extraction,
wall detection, and surface fitting together — one robust angle per pooled
batch.

Two density estimators are shown:

- :meth:`DensityEstimator.binning` (the default) — top-hat histogram
  with geometry-aware ``dV`` normalisation. Fast and exact; intrinsically
  noisy at low per-cell counts.
- :meth:`DensityEstimator.gaussian` — 3D Gaussian KDE on the cell
  centres. Smooth density field with no per-cell Poisson noise; the
  estimator of choice when running per-frame analyses or on systems
  with low atom density per cell.
"""

from wetting_angle_kit.analysis import (
    CoupledFit2DAnalyzer,
    DensityEstimator,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# --- Step 2: Identify water-oxygen atoms ---
wat_find = LammpsDumpWaterFinder(
    filename,
    oxygen_type=1,
    hydrogen_type=2,
)
oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 3: Define the grid ---
grid_params = {
    "xi_0": 0.0,
    "xi_f": 70.0,
    "dx": 2.0,
    "zi_0": 0.0,
    "zi_f": 70.0,
    "dz": 2.0,
}

# --- Step 4: Pick a density estimator ---
# Top-hat histogram on the sampling grid (default):
estimator = DensityEstimator.binning()
# Swap in the Gaussian KDE for smoother per-cell density. ``density_sigma``
# is the Gaussian kernel width; 3 Å is a sensible default for
# room-temperature water:
# estimator = DensityEstimator.gaussian(density_sigma=2.5)

# --- Step 5: Build the analyzer ---
analyzer = CoupledFit2DAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    grid_params=grid_params,
    density_estimator=estimator,
    # Pool 10 frames per batch — the coupled fit benefits from
    # statistics; ``batch_size=-1`` pools the entire trajectory.
    temporal_aggregator=TemporalAggregator(batch_size=10),
)

# --- Step 6: Run analysis on a frame range ---
# 20 frames at batch_size=10 gives two pooled batches.
results = analyzer.analyze(range(0, 20))
print("Mean contact angle (°):", results.mean_angle)
print("Std across batches (°):", results.std_angle)

# Per-batch detail:
batch = results.batches[0]
print(
    f"Frames {batch.frames[0]}–{batch.frames[-1]}: "
    f"angle = {batch.angle:.2f}°, "
    f"R_eq = {batch.model_params['R_eq']:.2f} Å, "
    f"z_wall = {batch.model_params['zi_0']:.2f} Å"
)
