Visualisation Tutorial — Angle Evolution and Density Contour
============================================================

Two trajectory-level plotters cover the most common visual outputs:

* :class:`AngleEvolutionPlotter` — per-batch contact angle vs time,
  with an optional ``±σ`` band and a cumulative running mean
  overlay. Works on any results object that exposes ``.batches``
  with ``.angle`` and ``.frames``.
* :class:`DensityContourPlotter` — 2D density field with the fitted
  spherical cap arc and wall line overlaid. Accepts a single batch
  or a full results object (averaged density); 3D results are
  azimuthally collapsed to the same 2D plane.

----

1. Angle evolution plot
-----------------------

The plotter takes a results object directly and exposes a
``.plot()`` method that returns a Plotly figure. The two key toggles
are ``per_frame_std`` (draws the inter-batch ``±σ`` band from
``angle_std``) and ``running_mean`` (overlays the cumulative running
mean with its own cumulative ``±σ`` band).

.. code-block:: python

   from wetting_angle_kit.analysis import (
       DensityEstimator,
       InterfaceExtractor,
       SpaceSampling,
       SurfaceFitter,
       TrajectoryAnalyzer,
       WallDetector,
   )
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.visualization import AngleEvolutionPlotter

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"
   oxygen_indices = LammpsDumpWaterFinder(
       filename, oxygen_type=1, hydrogen_type=2
   ).get_water_oxygen_indices(frame_index=0)

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor(
           sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
           density=DensityEstimator.gaussian(),
       ),
       surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
       temporal_aggregator=TemporalAggregator(batch_size=1),
   )
   results = analyzer.analyze(range(0, 24))

   plotter = AngleEvolutionPlotter(
       results,
       label="spherical_4k",
       timestep=0.5,  # 0.5 time units per dumped frame
       time_unit="ps",
   )
   fig = plotter.plot(per_frame_std=True, running_mean=True)
   fig.show()  # or fig.write_html("angle_evolution.html")

The figure has up to four traces per trajectory:

* the per-batch line (solid),
* the per-batch ``±σ`` band (filled, semi-transparent),
* the cumulative running mean (dashed),
* the cumulative ``±σ`` band of the running mean (filled).

Coupled-fit result objects don't carry ``angle_std`` per batch, so
the per-batch band is omitted; the running mean band is always
available.

The plotter also implements :class:`BaseTrajectoryPlotter` so
``plotter.summary()`` returns a list of :class:`TrajectoryStats`
with the mean angle, std, sample count, and a per-method surface area
(shoelace polygon area for slicing batches; spherical-cap segment
area for whole / coupled-fit batches).

----

2. Density contour plot
-----------------------

For a coupled-fit analysis, :class:`DensityContourPlotter` draws the
2D density grid with the fitted spherical cap arc (dashed) and wall
line (dotted) overlaid. Pass either a single batch result or a full
results object. The example below uses the default histogram
estimator; passing ``density_estimator=DensityEstimator.gaussian(...)``
on the analyzer constructor renders the contour over a smoothed
density field without touching anything else here:

.. code-block:: python

   from wetting_angle_kit.analysis import CoupledFit2DAnalyzer
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.visualization import DensityContourPlotter

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"
   oxygen_indices = LammpsDumpWaterFinder(
       filename, oxygen_type=1, hydrogen_type=2
   ).get_water_oxygen_indices(frame_index=0)

   coupled_fit = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params={
           "xi_0": 0.0,
           "xi_f": 70.0,
           "dx": 4.0,
           "zi_0": 0.0,
           "zi_f": 70.0,
           "dz": 4.0,
       },
       temporal_aggregator=TemporalAggregator(batch_size=10),
   )
   results = coupled_fit.analyze(range(0, 24))

   # One batch:
   DensityContourPlotter(results.batches[0], label="spherical_4k").plot().show()

   # Averaged across batches:
   DensityContourPlotter(results, label="spherical_4k").plot().show()

3D-results inputs are azimuthally averaged onto the same ``(r, z)``
plane before contouring, so the same plotter works for
:class:`CoupledFit3DResults` and
:class:`CoupledFit3DBatchResult`. The default title indicates the
azimuthal collapse so plots are unambiguous.

----

3. 3D isosurface visualisation
------------------------------

:class:`DensityContourPlotter` also supports interactive 3D
isosurface plots via its :meth:`plot_3d_isosurface` method. This
requires a **3D source** (:class:`CoupledFit3DResults` or
:class:`CoupledFit3DBatchResult`) because the full 3D density grid
is needed. The figure includes a density-threshold slider so you
can sweep through iso-levels, a semi-transparent wall plane, and
(optionally) the fitted sphere wireframe overlay.

Below is a complete example that runs the 3D coupled-fit analysis
and then generates both the azimuthally-averaged 2D contour and
the full 3D isosurface:

.. code-block:: python

   from wetting_angle_kit.analysis import CoupledFit3DAnalyzer
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.visualization import DensityContourPlotter

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"
   oxygen_indices = LammpsDumpWaterFinder(
       filename, oxygen_type=1, hydrogen_type=2
   ).get_water_oxygen_indices(frame_index=0)

   analyzer = CoupledFit3DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params={
           "xi_0": -40.0,
           "xi_f": 40.0,
           "dx": 4.0,
           "yi_0": -40.0,
           "yi_f": 40.0,
           "dy": 4.0,
           "zi_0": 0.0,
           "zi_f": 60.0,
           "dz": 4.0,
       },
       temporal_aggregator=TemporalAggregator(batch_size=-1),
   )
   results = analyzer.analyze(range(0, 24))

   plotter = DensityContourPlotter(results, label="4k spherical drop")

   # 2D contour (azimuthally averaged) — works for any source type.
   plotter.plot(save_path="density_contour_2d.html")
   print("Saved: density_contour_2d.html")

   # 3D isosurface with density-threshold slider (3D sources only).
   plotter.plot_3d_isosurface(n_levels=10, save_path="density_isosurface_3d.html")
   print("Saved: density_isosurface_3d.html")

The slider exposes ``n_levels`` iso-density thresholds; each
level is coloured consistently using the colorscale (Jet by
default) so you can visually identify which density value each
surface corresponds to. Set ``show_fit=False`` to hide the
wireframe sphere if you only want the raw density surface.

----

4. Tips
-------

- Pass ``title=...`` on either ``.plot()`` to override the default.
  The default density plot drops the list of frame indices from the
  title (which can be very long for pooled batches); pass a custom
  title if you want batch identification displayed.
- Use ``stat="median"`` on the constructor of
  :class:`AngleEvolutionPlotter` to plot the per-batch median across
  slices instead of the mean (slicing results only; ignored for
  other result types).
- The package follows a "one plotter per concern" pattern rather
  than per analyzer — pass any analyzer's results object to either
  plotter, and the plotter dispatches on the result type.
- For the 3D isosurface, using coarser grids (~25 bins per axis)
  with many pooled frames (``batch_size=-1``) gives the best
  visual result. Finer grids produce noisier surfaces without much
  benefit for visualisation.
