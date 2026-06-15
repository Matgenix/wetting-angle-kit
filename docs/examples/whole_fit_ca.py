"""Whole-shape fit contact-angle example.

Runs the whole-fit pipeline (full-sphere Fibonacci ray fan + algebraic
sphere fit + wall atoms from the trajectory) on a LAMMPS dump file,
with 100 bootstrap resamples for the angle uncertainty.
"""

from wetting_angle_kit.analysis import (
    DensityEstimator,
    InterfaceExtractor,
    SpaceSampling,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.parsers import (
    LammpsDumpParser,
    LammpsDumpWallParser,
    LammpsDumpWaterFinder,
)

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# --- Step 2: Identify water-oxygen and wall-atom indices ---
wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)

# Wall parser: ``liquid_particle_types`` lists the liquid types to EXCLUDE.
wall_parser = LammpsDumpWallParser(filename, liquid_particle_types=[1, 2])
carbon_indices = wall_parser.parse(frame_index=0)

# --- Step 3: Build the whole-fit analyzer ---
# Strategies: full-sphere Fibonacci ray fan + sphere fit + from_atoms wall.
analyzer = TrajectoryAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    interface_extractor=InterfaceExtractor(
        sampling=SpaceSampling.rays(n_rays_sphere=400),
        density=DensityEstimator.gaussian(density_sigma=3.0),
    ),
    surface_fitter=SurfaceFitter.whole(
        surface_filter_offset=3.0,
        bootstrap_samples=100,
    ),
    wall_detector=WallDetector.from_atoms(
        wall_atom_indices=carbon_indices,
        method="mean_top_layer",
        top_layer_tolerance=1.0,
    ),
    wall_atom_indices=carbon_indices,
)

# --- Step 4: Run the analysis ---
batch = analyzer.analyze([1]).batches[0]
print(f"angle = {batch.angle:.2f}° ± {batch.angle_std:.2f}° (bootstrap)")
print(f"R     = {batch.popt[3]:.2f} Å, z_wall = {batch.z_wall:.2f} Å")
print(f"rms residual on the shell = {batch.rms_residual:.2f} Å")
