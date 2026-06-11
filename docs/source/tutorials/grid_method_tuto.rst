Tutorial: Grid-Based Interface Extraction
==========================================

This tutorial covers the **grid-based interface extractors** —
:meth:`InterfaceExtractor.grid_gaussian` and
:meth:`InterfaceExtractor.grid_binning`. They are an alternative to
the ray-fan extractors used in the
:doc:`slicing_method_tuto` and
:doc:`whole_fit_tuto`: instead of locating the interface as the
half-density point of a 1D tanh fit along each ray, they build a
2D or 3D density grid and recover the interface as the iso-density
contour at the half-bulk level.

----

1. When to pick grid over rays?
-------------------------------

Both extractors plug into the same :class:`TrajectoryAnalyzer` and
produce the same downstream result objects, so the choice is mostly
about how the noise/cost trade-off lands on your system:

* **Ray fans** sample density along a small number of well-chosen
  directions; each ray's 1D tanh fit is cheap. Best when atom
  statistics per frame are high and the droplet has well-defined
  symmetry.
* **Grids** estimate density on every cell of a fixed mesh, then
  trace an iso-contour. Closer to the "average over many frames"
  intuition; the per-cell density gets smoother as more frames are
  pooled. Robust when individual frames are sparse, but it scales
  with the number of cells rather than the number of rays so the
  grid resolution matters more than in the ray case.

The grid extractors require ``scikit-image`` for the iso-contour
tracing (marching squares in 2D, marching cubes in 3D). Install via
the ``grid3d`` extra::

   pip install wetting-angle-kit[grid3d]

----

2. Worked example: ``grid_gaussian`` + slicing fit
---------------------------------------------------

A spherical droplet, with a 2D density grid in the
:math:`(r, z)` plane (the slicing-mode grid extractor projects atoms
to ``(r, z)`` via the droplet's radial symmetry — see
:doc:`../introduction/theoretical_foundations` section 4.2 for the
volume normalisation):

.. code-block:: python

   from wetting_angle_kit.analysis import (
       InterfaceExtractor,
       SurfaceFitter,
       TrajectoryAnalyzer,
       WallDetector,
   )
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"
   oxygen_indices = LammpsDumpWaterFinder(
       filename, oxygen_type=1, hydrogen_type=2
   ).get_water_oxygen_ids(frame_index=0)

   # 2D grid for slicing-mode extraction: (xi, zi) cells.
   # Aim for cells small enough to resolve the interface (~1 Å is plenty
   # for a Gaussian-smoothed grid) but large enough that occupied cells
   # carry many atoms.
   grid_params = {
       "xi_0": 0.0,
       "xi_f": 40.0,
       "nbins_xi": 26,
       "zi_0": 0.0,
       "zi_f": 40.0,
       "nbins_zi": 26,
   }

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor.grid_gaussian(
           grid_params=grid_params,
           density_sigma=3.0,
       ),
       surface_fitter=SurfaceFitter.slicing(surface_filter_offset=3.0),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
       temporal_aggregator=TemporalAggregator(batch_size=1),
   )
   batch = analyzer.analyze([1]).batches[0]
   print(
       f"Angle (grid_gaussian + slicing): {batch.angle:.2f}° " f"± {batch.angle_std:.2f}°"
   )

----

3. Histogram alternative: ``grid_binning``
------------------------------------------

Same flow but the density estimator is a 1D top-hat
(``rho = counts / dV``) rather than a Gaussian KDE. Numerically
cheaper but noisier per cell. Use a smaller grid to keep enough
atoms per cell:

.. code-block:: python

   from wetting_angle_kit.analysis import InterfaceExtractor

   grid_params = {
       "xi_0": 0.0,
       "xi_f": 40.0,
       "nbins_xi": 16,  # coarser
       "zi_0": 0.0,
       "zi_f": 40.0,
       "nbins_zi": 16,
   }

   extractor = InterfaceExtractor.grid_binning(grid_params=grid_params)
   # ... plug into TrajectoryAnalyzer exactly as above.

The slicing fitter's ``surface_filter_offset`` is a useful knob
here: histogram-based grids tend to have an iso-contour "floor"
just above the wall, which the filter drops out of the circle fit.

----

4. 3D iso-surface for the whole-fit
------------------------------------

The grid extractors also work in whole-fit mode for spherical
droplets — the 2D density grid is replaced by a 3D one, and the
half-bulk iso-surface is traced via marching cubes:

.. code-block:: python

   grid_params_3d = {
       "xi_0": -30.0,
       "xi_f": 30.0,
       "nbins_xi": 26,
       "yi_0": -30.0,
       "yi_f": 30.0,
       "nbins_yi": 26,
       "zi_0": 0.0,
       "zi_f": 35.0,
       "nbins_zi": 26,
   }

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor.grid_gaussian(
           grid_params=grid_params_3d,
           density_sigma=3.0,
       ),
       surface_fitter=SurfaceFitter.whole(
           surface_filter_offset=3.0,
           bootstrap_samples=100,
       ),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
   )
   batch = analyzer.analyze([1]).batches[0]
   print(
       f"Angle (grid_gaussian + whole-fit): "
       f"{batch.angle:.2f}° ± {batch.angle_std:.2f}°"
   )

Three notes on the 3D case:

* ``xi/yi`` are in the **droplet-centred frame** (the per-frame COM
  is subtracted before binning); ``zi`` stays in the lab frame so
  the wall position keeps its physical meaning.
* ``grid + whole-fit`` is currently spherical-only — cylindrical
  droplets need the ray-fan extractor because the centred-grid
  convention doesn't accommodate the cylinder axis spanning the
  full box. For cylinder whole fits use
  :meth:`InterfaceExtractor.rays_gaussian` with ``delta_cylinder``.
* Marching cubes can be slow on dense 3D grids; if performance
  matters, start with 20–30 bins per axis and only refine if the
  recovered angle is grid-resolution-limited.

----

5. Tips
-------

- **Grid bounds**: always pick ``xi_f``, ``yi_f``, ``zi_f`` so the
  full droplet fits comfortably inside the grid; the iso-contour
  tracer can't extrapolate.
- **Smoothing**: ``density_sigma`` on the Gaussian variant
  controls cell smoothing; values around the interface thickness
  (~1–3 Å for water) work well. The histogram variant exposes no
  smoothing knob — choose the cell size accordingly.
- **Cell size vs ``surface_filter_offset``**: rows of the 2D grid
  closest to the wall are normalised by a narrow annular volume
  (``2π r dr dz``) which inflates noise. The slicing fitter's
  ``surface_filter_offset=3.0`` (instead of the default 2.0)
  reliably drops the noisy floor.
- **Comparison plot**: it's often useful to run the same trajectory
  through both ``rays_gaussian`` and ``grid_gaussian`` and check
  the two angles agree within method-dependent tolerance (a few
  degrees on 4k-atom droplets). If they diverge more than ~5°, one
  of them is misconfigured (most often the grid bounds are too
  tight or ``surface_filter_offset`` is too small).
