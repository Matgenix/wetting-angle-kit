# Tutorial: Using the Parser Module

This tutorial shows how to load different trajectory formats using the `wetting_angle_kit.parser` submodule.

The parser provides a unified interface to read atomic coordinates from:
- LAMMPS dump files (`DumpParser`, `    DumpWaterMoleculeFinder`)
- ASE `.traj` files (`ASE_Parser`, `ASE_WaterMoleculeFinder`)
- XYZ files (`XYZParser`)

Each parser can extract atomic positions for selected frames and atoms, allowing efficient and flexible analysis of molecular simulations.

---

## 1. General Workflow

Every parser follows the same pattern:

1. Initialize the parser with your trajectory file.
2. (Optional) Use a `WaterMoleculeFinder` class to locate oxygen atoms belonging to water molecules.
3. Extract coordinates of all atoms or only selected indices using `.parse(frame_index, indices)`.

The `.parse()` method always returns a NumPy array of shape `(N, 3)` containing the atomic coordinates.

---

## 2. Example: LAMMPS Dump File
```python
from wetting_angle_kit.parser import DumpParser, DumpWaterMoleculeFinder

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

# --- Step 2: Initialize the water molecule finder ---
# Specify particle types for the wall and for water oxygens and hydrogens
wat_find = DumpWaterMoleculeFinder(
    filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2
)

# --- Step 3: Identify oxygen atoms for frame 0 ---
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 4: Initialize the parser ---
parser = DumpParser(filename)

# --- Step 5: Extract coordinates of only the water oxygens ---
oxygen_positions = parser.parse(frame_index=0, indices=oxygen_indices)
print("Extracted positions for", len(oxygen_positions), "oxygen atoms.")
```

**Notes:**
- Use `indices=None` to parse all atoms.
- `.parse()` returns NumPy coordinates for the selected frame.

---

## 3. Example: ASE Trajectory File
```python
from wetting_angle_kit.parser import ASE_Parser, ASE_WaterMoleculeFinder

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/slice_10_mace_mlips_cylindrical_2_5.traj"

# --- Step 2: Initialize the water molecule finder ---
wat_find = ASE_WaterMoleculeFinder(
    filename,
    particle_type_wall=["C"],  # Wall elements (e.g., carbon)
    oh_cutoff=0.4,  # O–H bond cutoff distance
)

# --- Step 3: Identify water oxygens for frame 0 ---
oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 4: Initialize the parser ---
parser = ASE_Parser(filename)

# --- Step 5: Extract oxygen atom positions only ---
oxygen_positions = parser.parse(frame_index=0, indices=oxygen_indices)
print("Extracted positions for", len(oxygen_positions), "oxygen atoms.")
```

**Tip:** The ASE parser works for any ASE-compatible trajectory (e.g., `.traj`, `.extxyz`, etc.).

---

## 4. Example: XYZ File

```python
from wetting_angle_kit.parser import XYZParser

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/slice_10_mace_mlips_cylindrical_2_5.xyz"

# --- Step 2: Initialize the parser ---
xyz_parser = XYZParser(filename)

# --- Step 3: Retrieve positions for the first frame ---
positions = xyz_parser.parse(frame_index=0)
print("Loaded frame with", len(positions), "atoms.")

# --- Step 4 (Optional): Parse only a subset of atoms ---
# For example, extract the first 50 atoms
subset_positions = xyz_parser.parse(frame_index=0, indices=list(range(50)))
print("Subset of 50 atoms extracted successfully.")
```

---

## 5. Summary

The parser module provides:
- **Unified interface** across LAMMPS, ASE, and XYZ formats
- **Selective parsing** using frame number and atom indices
- **Water molecule identification** to filter oxygen atoms from bulk water with tools from ase and ovito library

All parsers return NumPy arrays of shape `(N, 3)` for direct use in analysis pipelines.
