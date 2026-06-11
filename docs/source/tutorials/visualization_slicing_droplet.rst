Visualisation Tutorial — Per-Frame Droplet Snapshot
====================================================

This tutorial uses :class:`DropletSlicePlotter` to draw a single-frame
snapshot of a droplet, overlaying the recovered interface contour,
the fitted Kasa circle, the tangent at the contact point, and the
wall atom positions. The plotter takes raw arrays
(atom positions, surface points, fit parameters), so it works on the
output of the slicing pipeline once you pull the corresponding fields
off the result object.

----

1. Overview
-----------

The workflow:

1. Identify water molecules and read their positions for the frame.
2. Read the wall atom positions for the same frame.
3. Run the slicing pipeline (:class:`TrajectoryAnalyzer` with
   :class:`SurfaceFitter.slicing()`) on that frame; pull the
   ``slice_surfaces`` / ``slice_popts`` / ``per_slice_angles`` arrays
   off the resulting :class:`SlicingBatchResult`.
4. Pick one slice index and pass the corresponding entries to
   :class:`DropletSlicePlotter`.

----

2. Import Required Modules
---------------------------

.. code-block:: python

   import numpy as np

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

----

3. Define the Input Trajectory
-------------------------------

.. code-block:: python

   filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"
   frame_index = 10

----

4. Identify Water Molecules
----------------------------

.. code-block:: python

   wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
   oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
   print("Number of water molecules detected:", len(oxygen_indices))

----

5. Read Atom and Wall Positions
--------------------------------

.. code-block:: python

   parser = LammpsDumpParser(filepath=filename)
   oxygen_position = parser.parse(frame_index=frame_index, indices=oxygen_indices)

   # Wall parser: ``liquid_particle_types`` lists what to EXCLUDE
   # (the liquid), leaving the wall atoms.
   wall_parser = LammpsDumpWallParser(filename, liquid_particle_types=[1, 2])
   wall_coords = wall_parser.parse(frame_index=frame_index)

----

6. Run the Slicing Pipeline
----------------------------

.. code-block:: python

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
   print("Per-slice contact angles (°):", batch.per_slice_angles)

----

7. Visualise the Droplet
-------------------------

The plotter takes a single slice's data; pick a slice index and
pull the corresponding entries off the batch:

.. code-block:: python

   plotter = DropletSlicePlotter(center=True)

   slice_idx = 0  # pick a slice — all three arrays are parallel
   fig = plotter.plot_surface_points(
       oxygen_position=oxygen_position,
       surface_data=[batch.slice_surfaces[slice_idx]],
       popt=batch.slice_popts[slice_idx],
       wall_coords=wall_coords,
       alpha=float(batch.per_slice_angles[slice_idx]),
   )

   # Interactive view in a notebook
   fig.show()
   # Or save a standalone HTML page
   fig.write_html("droplet_plot.html")

The figure overlays four layers: the raw water-oxygen positions, the
recovered interface contour for the chosen slice, the fitted Kasa
circle, and the wall atoms. ``DropletSlicePlotter`` accepts
``center=True`` (default) to centre the plot on the droplet COM.
