"""Sliced contact-angle example.

Runs the per-frame sliced (circle-fitting) analyzer on a LAMMPS dump
file and prints the resulting mean contact angle.
"""

from wetting_angle_kit.contact_angle_method import contact_angle_analyzer
from wetting_angle_kit.parser import DumpParser, DumpWaterMoleculeFinder

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# --- Step 2: Identify the water molecules (oxygen-bonded-to-two-H) ---
wat_find = DumpWaterMoleculeFinder(
    filename,
    particle_type_wall={3},  # Wall atom types
    oxygen_type=1,
    hydrogen_type=2,
)

# `oxygen_indices` are LAMMPS particle IDs for the dump format.
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 3: Build the sliced analyzer ---
parser = DumpParser(filename)
analyzer = contact_angle_analyzer(
    method="sliced",
    parser=parser,
    output_dir="results_sliced_example",
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    delta_gamma=20,  # Azimuthal step for spherical slicing (degrees)
)

# --- Step 4: Run analysis for a frame range ---
results = analyzer.analyze([1])
print("Mean contact angle (°):", results["mean_angle"])
print("Frames analyzed:", results["frames_analyzed"])
