"""End-to-end example: slicing pipeline + per-frame droplet snapshot.

Runs the slicing-fit pipeline on a LAMMPS dump file, pulls one slice's
interface contour + fitted circle off the result, and renders the
droplet snapshot with :class:`DropletSlicePlotter`.
"""

from wetting_angle_kit.analysis import (
    InterfaceExtractor,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.parsers import (
    LammpsDumpParser,
    LammpsDumpWallParser,
    LammpsDumpWaterFinder,
)
from wetting_angle_kit.visualization import DropletSlicePlotter

# --- 1. Define the input trajectory ---
filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"
frame_index = 10

# --- 2. Identify water-oxygen atoms ---
wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
print("Number of water molecules detected:", len(oxygen_indices))

# --- 3. Read atom and wall positions for the frame ---
parser = LammpsDumpParser(filepath=filename)
oxygen_position = parser.parse(frame_index=frame_index, indices=oxygen_indices)

# Wall parser: ``liquid_particle_types`` lists what to EXCLUDE
# (the liquid), leaving the wall atoms.
wall_parser = LammpsDumpWallParser(filename, liquid_particle_types=[1, 2])
wall_coords = wall_parser.parse(frame_index=frame_index)

# --- 4. Run the slicing pipeline on the chosen frame ---
analyzer = TrajectoryAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="cylinder_y",
    interface_extractor=InterfaceExtractor.rays_gaussian(
        delta_cylinder=5.0,
        delta_polar=8.0,
    ),
    surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
    wall_detector=WallDetector.min_plus_offset(offset=0.0),
    temporal_aggregator=TemporalAggregator(batch_size=1),
)
batch = analyzer.analyze([frame_index]).batches[0]
print("Per-slice contact angles (°):", batch.per_slice_angles.tolist())

# --- 5. Visualise one slice ---
plotter = DropletSlicePlotter(center=True)
slice_idx = 0  # any 0..len(slice_surfaces)-1

fig = plotter.plot_surface_points(
    oxygen_position=oxygen_position,
    surface_data=[batch.slice_surfaces[slice_idx]],
    popt=batch.slice_popts[slice_idx],
    wall_coords=wall_coords,
    alpha=float(batch.per_slice_angles[slice_idx]),
)

fig.write_html("droplet_plot.html")
print("Plot saved as 'droplet_plot.html'")
