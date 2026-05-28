"""
Example: Using LammpsDumpParser and LammpsDumpWaterFinder

This example shows how to:
1. Identify water molecules in a LAMMPS dump file.
2. Extract only their oxygen atom coordinates.
"""

from wetting_angle_kit.parsers import (
    AseParser,
    AseWaterFinder,
    LammpsDumpParser,
    LammpsDumpWaterFinder,
    XYZParser,
)

# --- Define input file ---
filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

# --- Initialize water molecule finder ---
wat_find = LammpsDumpWaterFinder(
    filename,
    particle_type_wall={3},  # atom type for wall
    oxygen_type=1,  # atom type for oxygen
    hydrogen_type=2,  # atom type for hydrogen
)

# --- Identify water oxygen indices for the first frame ---
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print(f"Number of water molecules: {len(oxygen_indices)}")

# --- Initialize parser ---
parser = LammpsDumpParser(filename)

# --- Extract only oxygen coordinates for frame 0 ---
# For LammpsDumpParser, `indices` are LAMMPS particle IDs (because LAMMPS may
# reorder atoms between frames). For XYZParser/AseParser, `indices` are
# 0-based positional indices.
oxygen_positions = parser.parse(frame_index=0, indices=oxygen_indices)
print("Extracted oxygen coordinates shape:", oxygen_positions.shape)

# --- Optional: Extract all atoms ---
# all_positions = parser.parse(frame_index=0)
# print("All atom positions shape:", all_positions.shape)

"""
Example: Using AseParser and AseWaterFinder

This example demonstrates how to:
1. Identify water oxygens in an ASE trajectory.
2. Extract their positions for a given frame.
"""

# --- Define input file ---
filename = "../../tests/trajectories/slice_10_mace_mlips_cylindrical_2_5.traj"

# --- Initialize water molecule finder ---
wat_find = AseWaterFinder(
    filename,
    particle_type_wall=["C"],  # element name for wall
    oh_cutoff=1.2,  # O–H bond cutoff (Å); ASE NeighborList handles the
    # per-atom splitting internally now.
)

# --- Get oxygen indices for frame 0 ---
oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)
print(f"Number of water molecules: {len(oxygen_indices)}")

# --- Initialize parser ---
parser = AseParser(filename)

# --- Extract oxygen coordinates only ---
oxygen_positions = parser.parse(frame_index=0, indices=oxygen_indices)
print("Extracted oxygen coordinates shape:", oxygen_positions.shape)

"""
Example: Using XYZParser

This example demonstrates how to:
1. Load atomic positions from an XYZ file.
2. Extract all atoms or a subset of atoms.
"""

# --- Define input file ---
filename = "../../tests/trajectories/slice_10_mace_mlips_cylindrical_2_5.xyz"

# --- Initialize parser ---
xyz_parser = XYZParser(filename)

# --- Extract all atom coordinates for frame 0 ---
positions = xyz_parser.parse(frame_index=0)
print("Total atoms loaded:", len(positions))

# --- Extract subset of atoms (first 50) ---
subset = xyz_parser.parse(frame_index=0, indices=list(range(50)))
print("Subset (50 atoms) shape:", subset.shape)
