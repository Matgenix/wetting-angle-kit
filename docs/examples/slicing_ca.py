"""Slicing-pipeline contact-angle example.

Runs the per-frame slicing-fit pipeline (ray-fan extractor + algebraic
circle fitter + interface-derived wall) on a LAMMPS dump file and prints
the recovered mean contact angle.
"""

from wetting_angle_kit.analysis import (
    InterfaceExtractor,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# --- Step 2: Identify the water-oxygen atoms ---
wat_find = LammpsDumpWaterFinder(
    filename,
    oxygen_type=1,
    hydrogen_type=2,
)
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 3: Build the trajectory analyzer ---
# Strategies: ray-fan Gaussian extractor + slicing fitter +
# interface-derived wall + per-frame batching.
analyzer = TrajectoryAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    interface_extractor=InterfaceExtractor.rays_gaussian(
        delta_azimuthal=20.0,  # 20° between slicing planes
        delta_polar=8.0,  # 8° in-plane ray step
    ),
    surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
    wall_detector=WallDetector.min_plus_offset(offset=0.0),
    temporal_aggregator=TemporalAggregator(batch_size=1),
)

# --- Step 4: Run analysis on a frame range ---
results = analyzer.analyze([1])
print("Mean contact angle (°):", results.mean_angle)
print("Std across batches (°):", results.std_angle)

# Per-batch detail:
batch = results.batches[0]
print(
    f"Frame {batch.frames[0]}: angle = {batch.angle:.2f}°, "
    f"per-slice σ = {batch.angle_std:.2f}°, "
    f"rms residual = {batch.rms_residual:.2f} Å"
)
