"""Coupled-binning contact-angle example.

Runs the joint hyperbolic-tangent fit on a 2D binned density grid via
:class:`CoupledBinning2DAnalyzer`. One angle per pooled batch — best
when you have many frames per batch.
"""

from wetting_angle_kit.analysis import CoupledBinning2DAnalyzer
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
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 3: Define the binning grid ---
binning_params = {
    "xi_0": 0.0,
    "xi_f": 70.0,
    "nbins_xi": 30,
    "zi_0": 0.0,
    "zi_f": 70.0,
    "nbins_zi": 30,
}

# --- Step 4: Build the analyzer ---
analyzer = CoupledBinning2DAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    binning_params=binning_params,
    # Pool 10 frames per batch (legacy split_factor=10 analog).
    temporal_aggregator=TemporalAggregator(batch_size=10),
)

# --- Step 5: Run analysis on a frame range ---
results = analyzer.analyze([1])
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
