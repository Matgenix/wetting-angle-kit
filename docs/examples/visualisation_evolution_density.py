"""End-to-end example: angle evolution + density contour plots.

Runs both the per-frame slicing pipeline and the coupled-binning
analyzer on the same trajectory, then renders the two trajectory-level
plots: the angle evolution curve (with per-batch ±σ band and running
mean) and the density contour with the fitted spherical cap overlaid.
"""

from wetting_angle_kit.analysis import (
    CoupledBinning2DAnalyzer,
    InterfaceExtractor,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
from wetting_angle_kit.visualization import (
    AngleEvolutionPlotter,
    DensityContourPlotter,
)

filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# Water-oxygen atoms.
wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)

# --- 1. Slicing pipeline → angle evolution figure ---
slicing = TrajectoryAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    interface_extractor=InterfaceExtractor.rays_gaussian(
        delta_azimuthal=20.0, delta_polar=8.0
    ),
    surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
    wall_detector=WallDetector.min_plus_offset(offset=0.0),
    temporal_aggregator=TemporalAggregator(batch_size=1),
)
slicing_results = slicing.analyze(range(0, 50))

splot = AngleEvolutionPlotter(
    slicing_results,
    label="spherical_4k",
    timestep=0.5,
    time_unit="ps",
)
fig_evolution = splot.plot(per_frame_std=True, running_mean=True)
fig_evolution.write_html("angle_evolution.html")
print("Saved angle_evolution.html")

# --- 2. Coupled-binning analyzer → density contour figure ---
binning = CoupledBinning2DAnalyzer(
    parser=LammpsDumpParser(filename),
    atom_indices=oxygen_indices,
    droplet_geometry="spherical",
    binning_params={
        "xi_0": 0.0,
        "xi_f": 70.0,
        "bin_width_x": 2.0,
        "zi_0": 0.0,
        "zi_f": 70.0,
        "bin_width_z": 2.0,
    },
    temporal_aggregator=TemporalAggregator(batch_size=10),
)
binning_results = binning.analyze(range(0, 100))

# Pick the first batch (or pass ``binning_results`` directly to average
# the density across all batches before contouring).
bplot = DensityContourPlotter(binning_results.batches[0], label="spherical_4k")
fig_density = bplot.plot()
fig_density.write_html("density_contour.html")
print("Saved density_contour.html")
