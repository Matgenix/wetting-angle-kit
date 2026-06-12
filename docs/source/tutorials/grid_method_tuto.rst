Tutorial: Grid-Based Interface Extraction
==========================================

This tutorial covers the **grid-based interface extractors** —
:meth:`InterfaceExtractor.grid_gaussian` and
:meth:`InterfaceExtractor.grid_binning`. They are an alternative to
the ray-fan extractors used in the
:doc:`slicing_method_tuto` and
:doc:`whole_fit_tuto`: instead of locating the interface as the
half-density point of a 1D tanh fit along each ray, they evaluate a
density at each cell of a fixed grid and recover the interface as
the iso-density contour at the half-bulk level.

In slicing mode both grid extractors iterate per slice — per
azimuthal angle for spherical droplets, per axial step for cylinder
droplets — so the downstream :class:`SurfaceFitter.slicing` sees one
contour per slice and can expose per-slice asymmetry, exactly like
the rays variants.

----

1. When to pick grid over rays?
-------------------------------

Both extractors plug into the same :class:`TrajectoryAnalyzer` and
produce the same downstream result objects, so the choice is mostly
about how the noise/cost trade-off lands on your system:

* **Ray fans** sample density along a small number of well-chosen
  directions; each ray's 1D tanh fit is cheap. Best when atom
  statistics per frame are high and you want sub-frame resolution.
* **Grids** estimate density on every cell of a fixed mesh, then
  trace an iso-contour. Closer to the "average over many frames"
  intuition; the per-cell density gets smoother as more frames are
  pooled.

The grid extractors require ``scikit-image`` for the iso-contour
tracing (marching squares in 2D, marching cubes in 3D). Install via
the ``grid3d`` extra::

   pip install wetting-angle-kit[grid3d]

----

2. Worked example: ``grid_gaussian`` + slicing fit
---------------------------------------------------

A spherical droplet, with per-azimuthal-slice 2D density grids in the
``(s, z)`` plane — same density estimator as ``rays_gaussian``, just
sampled on a fixed grid rather than along rays:

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

   # 2D grid for each slice plane: ``s`` (in-plane radial) spans
   # ``[xi_0, xi_f]`` symmetrically around the slice centre to cover
   # the full diameter; ``z`` stays in the lab frame.
   grid_params = {
       "xi_0": -40.0,
       "xi_f": 40.0,
       "bin_width_x": 3.0,  # 3 Å cells in s
       "zi_0": 0.0,
       "zi_f": 40.0,
       "bin_width_z": 1.6,  # 1.6 Å cells in z
   }

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor.grid_gaussian(
           grid_params=grid_params,
           delta_azimuthal=20.0,  # 9 azimuthal slices
           density_sigma=2.0,
       ),
       surface_fitter=SurfaceFitter.slicing(surface_filter_offset=3.0),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
       temporal_aggregator=TemporalAggregator(batch_size=1),
   )
   batch = analyzer.analyze([1]).batches[0]
   print(
       f"Angle (grid_gaussian + slicing): {batch.angle:.2f}° "
       f"± {batch.angle_std:.2f}° across {len(batch.per_slice_angles)} slices"
   )

----

3. Histogram alternative: ``grid_binning``
------------------------------------------

Same per-slice iteration, but the density estimator is a top-hat
histogram of atoms within the slab ``|perp| ≤ bin_width_x / 2`` of
the slice plane. Numerically cheaper than the KDE; intrinsically
noisier because only atoms in the slab contribute (not all atoms
along the slice direction the way they do for ``rays_binning``).
Use coarser cells (thicker slab) than for ``grid_gaussian``:

.. code-block:: python

   from wetting_angle_kit.analysis import InterfaceExtractor

   grid_params = {
       "xi_0": -40.0,
       "xi_f": 40.0,
       "bin_width_x": 8.0,  # thick slab
       "zi_0": 0.0,
       "zi_f": 40.0,
       "bin_width_z": 3.0,
   }
   extractor = InterfaceExtractor.grid_binning(
       grid_params=grid_params,
       delta_azimuthal=60.0,  # fewer slices → more atoms per slab
   )

The slab thickness perpendicular to each slice plane is
``bin_width_x``, so refining the in-plane grid also thins the slab.
For systems with limited atom statistics per slab, the answer is
either coarser cells or fewer slices, not a finer grid.

----

4. 3D iso-surface for the whole-fit
------------------------------------

The grid extractors also work in whole-fit mode for spherical
droplets — the 2D density grid is replaced by a 3D one, and the
half-bulk iso-surface is traced via marching cubes. Whole mode
takes no ``delta_azimuthal`` / ``delta_cylinder``:

.. code-block:: python

   grid_params_3d = {
       "xi_0": -30.0,
       "xi_f": 30.0,
       "bin_width_x": 2.5,
       "yi_0": -30.0,
       "yi_f": 30.0,
       "bin_width_y": 2.5,
       "zi_0": 0.0,
       "zi_f": 35.0,
       "bin_width_z": 2.0,
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
* ``grid + whole-fit`` works for both spherical and cylinder
  droplets. For a cylinder, the user must pick ``yi_0`` / ``yi_f``
  to span the full cylinder axis (typically ``[-box_y / 2, +box_y /
  2]``); the centred-grid convention puts the droplet COM at the
  midpoint along ``y``, so a symmetric range covers the whole
  ridge. The fitter projects the 3D shell onto the ``(x, z)`` plane
  and does a 2D circle fit by translational invariance along
  ``y``.
* Marching cubes can be slow on dense 3D grids; if performance
  matters, start with 2–3 Å cells and only refine if the recovered
  angle is grid-resolution-limited.

----

5. Tips
-------

- **Grid bounds**: pick ``xi_f``, ``yi_f``, ``zi_f`` so the full
  droplet fits comfortably inside the grid (signed ``xi_0`` for the
  slicing case so the slice spans the full diameter). The
  iso-contour tracer can't extrapolate.
- **Cell sizes**: ``bin_width_x`` controls in-plane horizontal
  resolution; ``bin_width_z`` controls vertical. The range bounds
  are honoured exactly and the cell width is rounded to fit, so the
  effective cell size may differ slightly from the value you pass.
- **Comparison plot**: run the same trajectory through both
  ``rays_gaussian`` and ``grid_gaussian`` and check the two angles
  agree within method-dependent tolerance (a few degrees on
  4k-atom droplets). If they diverge by more than ~8°, one of them
  is misconfigured (most often the grid bounds are too tight or
  ``surface_filter_offset`` is too small).
- **grid_binning slab thickness**: the slab perpendicular to each
  slice equals ``bin_width_x``. If you see a noisy iso-contour,
  thicken it (larger ``bin_width_x``) before reaching for ``grid_gaussian``.
