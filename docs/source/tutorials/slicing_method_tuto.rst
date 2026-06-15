Tutorial: Contact Angle Analysis (Slicing Pipeline)
===================================================

This tutorial walks through the **slicing pipeline** built from the
strategy components of :class:`TrajectoryAnalyzer`: a ray-fan
interface extractor, a per-slice algebraic-circle fitter, and an
interface-derived wall detector. The slicing pipeline is the right
choice when you want a per-frame angle trace plus a sense of the
spread across slices.

----

1. Overview
-----------

The pipeline does three things per batch:

1. **Interface extraction.** The droplet is divided into vertical
   slicing planes (azimuthal slices for a spherical droplet,
   ``y``-step slices for a cylindrical droplet). Inside each plane a
   2D ray fan emits rays from the droplet centre of mass and locates
   the interface along each ray as the half-density point of a 1D
   tanh fit on the local density profile (Gaussian KDE by default).
2. **Wall detection.** The wall plane z-coordinate is taken as the
   minimum z over all interface points, plus a user-supplied offset
   (``min_plus_offset(offset=0)`` for the bare baseline).
3. **Surface fit.** An algebraic Taubin circle is fit to each slice's
   interface points after filtering out points within
   ``surface_filter_offset`` of the wall. The contact angle on each
   slice is the angle of intersection of that circle with the wall
   line; the batch's reported angle is the mean across slices, and
   :attr:`SlicingBatchResult.angle_std` is the empirical std.

----

2. Requirements
---------------

Before running the example, ensure you have installed the package
with the ovito extra (for LAMMPS dump files):

.. code-block:: bash

   pip install wetting-angle-kit[ovito]
   # (and the OVITO package itself via conda — see installation page)

Example trajectory::

   tests/trajectories/traj_spherical_drop_4k.lammpstrj

----

3. Example Code
---------------

.. code-block:: python

   from wetting_angle_kit.analysis import (
       InterfaceExtractor,
       SurfaceFitter,
       TrajectoryAnalyzer,
       WallDetector,
   )
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

   # --- Step 1: Define the trajectory file ---
   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

   # --- Step 2: Identify water-oxygen atoms ---
   wat_find = LammpsDumpWaterFinder(
       filename,
       oxygen_type=1,
       hydrogen_type=2,
   )
   oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
   print("Number of water molecules:", len(oxygen_indices))

   # --- Step 3: Build the trajectory analyzer ---
   # Strategies: rays extractor (Gaussian) + slicing fitter +
   # interface-derived wall + per-frame batching.
   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor(
           sampling=SpaceSampling.rays(
               delta_azimuthal=20.0,  # 20° between slicing planes
               delta_polar=8.0,  # 8° in-plane ray step
           ),
           density=DensityEstimator.gaussian(),
       ),
       surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
       temporal_aggregator=TemporalAggregator(batch_size=1),  # one angle per frame
   )

   # --- Step 4: Run the analysis on a frame range ---
   results = analyzer.analyze(range(0, 50))

   # --- Step 5: Inspect the results ---
   print("Mean contact angle (°):", results.mean_angle)
   print("Std across batches (°):", results.std_angle)
   for batch in results.batches[:3]:
       print(
           f"Frame {batch.frames[0]}: "
           f"angle = {batch.angle:.2f}°, "
           f"per-slice σ = {batch.angle_std:.2f}°, "
           f"rms residual = {batch.rms_residual:.2f} Å"
       )

----

4. Expected Output
------------------

On the water/graphene fixture above, single-frame output looks like::

   Number of water molecules: 1320
   Mean contact angle (°): 95.16
   Std across batches (°): 0.0
   Frame 0: angle = 95.16°, per-slice σ = 1.86°, rms residual = 0.45 Å

``std_angle`` is 0 here because only one batch was requested; pass a
multi-frame range to see the spread across batches.

The returned :class:`TrajectoryResults` object holds a list of
:class:`SlicingBatchResult` entries (one per batch). Each batch
carries:

* ``angle`` — mean contact angle across slices (°).
* ``angle_std`` — empirical standard deviation across slices (°).
* ``per_slice_angles`` — array of per-slice angles.
* ``slice_surfaces`` / ``slice_popts`` — per-slice interface points
  and fitted circle parameters (for plotting; see
  :doc:`visualization_slicing_droplet`).
* ``z_wall`` — wall position used by the fitter.
* ``rms_residual`` — mean of per-slice circle-fit RMS residuals (Å).

----

5. Tips
-------

- **Slicing step** (``delta_azimuthal`` for spherical droplets,
  ``delta_cylinder`` for cylinders): smaller step → more slices,
  more detail per batch, more cost. The default 20° gives 9 slices
  for a spherical droplet, plenty for a stable mean.
- **In-plane ray step** (``delta_polar``, both geometries): smaller
  step → more rays per slice, denser interface contour, more cost.
- **Wall offset** (``WallDetector.min_plus_offset(offset=O)``):
  raise ``O`` if the interface-derived baseline lands slightly into
  the wall layer (visible as inflated angles).
- **Surface filter offset**
  (``SurfaceFitter.slicing(surface_filter_offset=...)``): excludes
  interface points within this distance of the wall before the
  circle fit. Raise it if the wall-adjacent density is distorted by
  layering.
- **Cylindrical droplets**: pass ``droplet_geometry="cylinder_y"``
  (or ``"cylinder_x"``) and configure ``delta_cylinder`` instead of
  ``delta_azimuthal`` on the extractor.

For a side-by-side plot of the recovered interface and the fitted
circle, see :doc:`visualization_slicing_droplet`.

----

6. Alternative configurations
-----------------------------

6.1 Cylindrical droplets
^^^^^^^^^^^^^^^^^^^^^^^^

For a cylindrical droplet (e.g. water on a periodic stripe), swap
``delta_azimuthal`` for ``delta_cylinder`` (the step along the
cylinder axis) and tell the analyzer which axis the cylinder runs
along. Pick ``"cylinder_y"`` if the periodic ridge spans the box
along the lab-frame ``y`` axis; pick ``"cylinder_x"`` if it spans
along ``x``. The package handles ``cylinder_x`` by applying a
self-inverse ``x↔y`` column swap at the parser/analyzer boundary so
all downstream code can assume the cylinder axis is ``y`` —
analysis logic isn't duplicated between the two cases. Picking the
wrong axis is the cylinder analogue of confusing the in-plane
radial direction with the symmetry axis; symptoms are slicing
planes that go across the ridge (almost no atoms per slice) and a
fitter that either NaNs out or returns a non-physical angle:

.. code-block:: python

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="cylinder_y",  # or "cylinder_x"
       interface_extractor=InterfaceExtractor(
           sampling=SpaceSampling.rays(
               delta_cylinder=5.0,  # 5 Å between slicing planes
               delta_polar=8.0,
           ),
           density=DensityEstimator.gaussian(),
       ),
       surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
       temporal_aggregator=TemporalAggregator(batch_size=1),
   )

The mechanics are identical to the spherical case — same Taubin
circle fit per slice, same cap-angle formula — but slices step
along the cylinder axis rather than rotating azimuthally. The
fixture ``tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj``
in the repository is a cylindrical-droplet trajectory you can use
as a worked example.

6.2 ``rays`` (binning) alternative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same ray-fan geometry is available with a 1D histogram density
estimator instead of the Gaussian KDE. Use it when you want a
hard-cutoff per-sample density (fast, no smoothing parameter beyond
the bin width):

.. code-block:: python

   interface_extractor = InterfaceExtractor(
       sampling=SpaceSampling.rays(
           delta_azimuthal=20.0,
           delta_polar=8.0,
           points_per_angstrom=1.0,
       ),
       density=DensityEstimator.binning(bin_width=3.0),  # 3 Å diameter top-hat
   )

The ``bin_width`` parameter sets the diameter of the 3D top-hat
counted at each sample point along the ray; matching it to the
interface thickness (~1–3 Å for water) keeps the tanh fit
well-conditioned. Numerically the bin width plays the same role
``density_sigma`` plays for ``rays`` (Gaussian).

6.3 Pooled batches
^^^^^^^^^^^^^^^^^^

Replace ``batch_size=1`` with ``batch_size=N`` to pool
:math:`N` consecutive frames per fit — fewer batches, more atoms
per fit, less per-angle noise but no within-batch time resolution.
``batch_size=-1`` pools all requested frames into a single batch
(one angle for the whole trajectory).

.. code-block:: python

   temporal_aggregator = TemporalAggregator(batch_size=5)

.. note::

   With ``batch_size > 1``, the temporal aggregator pools
   **atom positions** across frames (after per-frame PBC recentring)
   before the extractor runs. The slicing pipeline then operates on
   a single density field built from the union of frames, giving one
   angle per batch with ``angle_std`` reflecting the spatial
   asymmetry of the *pooled* density — not per-frame variability.
   This is the right tool if you want a robust single angle over a
   steady-state window, with the per-slice scatter as an asymmetry
   diagnostic.

   If you want per-frame angles plus their across-frame mean and
   standard error, use ``batch_size=1`` and aggregate the angles
   yourself from the returned ``per_batch_angles`` array. The two
   modes are statistically different: pooled-atoms averages the
   density before measuring; pooled-angles measures each frame and
   then averages.

   Two subtle caveats of pooled-atoms mode: translational drift
   across the batch is handled (per-frame PBC recentring), but
   rotational drift and shape oscillations are smeared together
   with the spatial asymmetry. For steady-state droplets this is
   harmless; for transient regimes (wetting, dewetting, vibration)
   ``batch_size=1`` is the correct choice.

For physical context on the trade-off see
:doc:`../introduction/theoretical_foundations` section 9.

6.4 Grid alternative
^^^^^^^^^^^^^^^^^^^^

The grid extractors (:meth:`InterfaceExtractor.grid` (Gaussian) and
:meth:`InterfaceExtractor.grid` (binning)) pair with the slicing fitter exactly the same
way and are covered in :doc:`grid_method_tuto`. Use them when
ray-fan sampling is too sparse to resolve the interface.
