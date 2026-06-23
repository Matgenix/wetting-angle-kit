"""Coupled-fit 3D contact-angle example.

Runs the 3D coupled hyperbolic-tangent fit on a full ``(xi, yi, zi)``
density grid via :class:`CoupledFit3DAnalyzer`. The nine-parameter
model fits the spherical-cap interface and the wall plane
simultaneously, recovering a single robust angle per pooled batch.

Only spherical droplets are supported — cylindrical droplets reduce to
the 2D coupled fit by translational symmetry.
"""

from wetting_angle_kit.analysis import (
    CoupledFit3DAnalyzer,
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

# --- Step 3: Define the 3D grid ---
# xi/yi are in the droplet-centred frame; zi is in the lab frame so
# the wall position retains physical meaning.
grid_params = {
    "xi_0": -30.0,
    "xi_f": 30.0,
    "dx": 3.2,
    "yi_0": -30.0,
    "yi_f": 30.0,
    "dy": 3.2,
    "zi_0": 0.0,
    "zi_f": 60.0,
    "dz": 4.0,
}

# --- Step 4: Pick a density estimator ---
# Top-hat histogram on the 3D sampling grid (default):
estimator = DensityEstimator.binning()
# Swap in the Gaussian KDE for smoother per-cell density:
# estimator = DensityEstimator.gaussian(density_sigma=3.0)

# --- Step 5: Build the analyzer ---
analyzer = CoupledFit3DAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    grid_params=grid_params,
    density_estimator=estimator,
    # Pool all frames into a single batch — the 3D density needs more
    # atoms than the 2D one for comparable per-cell noise.
    temporal_aggregator=TemporalAggregator(batch_size=-1),
)

# --- Step 6: Run analysis ---
n_frames = LammpsDumpParser(filename).frame_count()
results = analyzer.analyze(range(0, n_frames))
print("Mean contact angle (°):", results.mean_angle)

# Per-batch detail:
batch = results.batches[0]
print(
    f"Frames {batch.frames[0]}–{batch.frames[-1]}: "
    f"angle = {batch.angle:.2f}°, "
    f"R_eq = {batch.model_params['R_eq']:.2f} Å, "
    f"z_wall = {batch.model_params['zi_0']:.2f} Å"
)
print(
    f"Droplet centre: "
    f"xi_c = {batch.model_params['xi_c']:.2f} Å, "
    f"yi_c = {batch.model_params['yi_c']:.2f} Å, "
    f"zi_c = {batch.model_params['zi_c']:.2f} Å"
)
