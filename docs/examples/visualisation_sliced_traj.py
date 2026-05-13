"""End-to-end example: sliced contact-angle pipeline plus visualization.

Run a single-frame sliced analysis on a LAMMPS dump file and save a PNG of
the droplet with the fitted circle, surface contour, and tangent at the
contact point.
"""

import numpy as np
from wetting_angle_kit.contact_angle_methods.sliced import ContactAngleSliced
from wetting_angle_kit.parsers import (
    LammpsDumpParser,
    LammpsDumpWallParser,
    LammpsDumpWaterFinder,
)
from wetting_angle_kit.visualization import DropletSlicePlotter

# --- 1. Define the Input Trajectory ---
# Adjust this to point to your local .lammpstrj file.
filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

# --- 2. Identify Water Molecules ---
wat_find = LammpsDumpWaterFinder(
    filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2
)

oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules detected:", len(oxygen_indices))

# --- 3. Parse Atomic Coordinates ---
parser = LammpsDumpParser(filepath=filename)
oxygen_position = parser.parse(frame_index=10, indices=oxygen_indices)

# Wall particles are everything not in the liquid types.
coord_wall = LammpsDumpWallParser(filename, liquid_particle_types=[1, 2])
wall_coords = coord_wall.parse(frame_index=10)

# --- 4. Compute Contact Angles ---
processor = ContactAngleSliced(
    liquid_coordinates=oxygen_position,
    liquid_geom_center=np.mean(oxygen_position, axis=0),
    droplet_geometry="cylinder_y",
    delta_cylinder=5,
    max_dist=100,
    width_cylinder=21,
)

list_alfas, array_surfaces, array_popt = processor.predict_contact_angle()
print("Per-slice contact angles (°):", list_alfas)

# --- 5. Visualize the Droplet ---
plotter = DropletSlicePlotter(center=True, show_wall=True, molecule_view=True)

plotter.plot_surface_points(
    oxygen_position=oxygen_position,
    surface_data=array_surfaces,
    popt=array_popt[0],
    wall_coords=wall_coords,
    output_filename="droplet_plot.png",
    alpha=list_alfas[0],
)

print("Plot saved as 'droplet_plot.png'")
