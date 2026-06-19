Tutorial: Contact Angle Analysis (Coupled Fit, 2D)
==================================================

This tutorial covers :class:`CoupledFit2DAnalyzer`, the
coupled-fit alternative to the composable
:class:`TrajectoryAnalyzer` pipeline. The analyzer solves interface
extraction, wall detection, and surface fit together by fitting a
seven-parameter hyperbolic-tangent density model directly to a 2D
density field on a fixed grid. One robust angle per pooled batch —
ideal when you have many frames and don't need per-frame time
resolution.

The per-cell density is computed by a pluggable
:class:`DensityEstimator` strategy: the default
:meth:`DensityEstimator.binning` is a top-hat histogram;
:meth:`DensityEstimator.gaussian` evaluates a 3D Gaussian KDE at
the cell centres for a smooth, Poisson-noise-free density — useful
for per-frame analyses where the histogram occasionally collapses
to a degenerate fit. See §6.2 for a worked example.

----

1. Overview
-----------

The pipeline does three things per batch:

1. **Density grid.** Pool the liquid atom positions across the
   batch's frames, project them to the ``(xi, zi)`` plane via the
   droplet's symmetry (radial for spherical droplets, perpendicular
   to the cylinder axis for cylindrical droplets), and build a 2D
   histogram with the user-supplied grid bounds and bin counts.
   Apply geometry-aware volume normalisation
   (``dV = 2π xi dxi dzi`` for spherical, ``dV = box_y · dxi dzi``
   for cylinder).
2. **NLLS fit.** Fit a seven-parameter hyperbolic-tangent
   density model

   .. math::

      \rho(\xi, z) = g(r) \cdot h(z - z_0),
      \qquad r = \sqrt{\xi^2 + (z - z_c)^2},

   with ``g(r)`` a radial sigmoid centred at ``(0, z_c)`` of
   equivalent radius ``R_{eq}`` and interface thickness ``t_1``, and
   ``h(z - z_0)`` a vertical sigmoid above the wall ``z_0`` of
   thickness ``t_2``. The fit returns the parameters
   ``(rho1, rho2, R_eq, z_c, z_0, t1, t2)``.
3. **Contact angle.** The cap–wall intersection geometry gives the
   contact angle directly:

   .. math::

      \cos \theta = \frac{z_0 - z_c}{R_{eq}}.

----

2. Prerequisites
----------------

Your trajectory file should contain atom IDs, types, and positions.
Example trajectory::

   tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj

----

3. Example Code
---------------

.. code-block:: python

   from wetting_angle_kit.analysis import CoupledFit2DAnalyzer
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

   # --- Step 1: Define the trajectory file ---
   filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

   # --- Step 2: Identify water-oxygen atoms ---
   wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
   oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)
   print("Number of water molecules:", len(oxygen_indices))

   # --- Step 3: Define the 2D sampling grid ---
   grid_params = {
       "xi_0": 0.0,
       "xi_f": 100.0,
       "dx": 2.0,
       "zi_0": 0.0,
       "zi_f": 100.0,
       "dz": 4.0,
   }

   # --- Step 4: Build the analyzer ---
   analyzer = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="cylinder_y",
       grid_params=grid_params,
       # 10-frame pooled batches
       temporal_aggregator=TemporalAggregator(batch_size=10),
   )

   # --- Step 5: Run analysis for a frame range ---
   results = analyzer.analyze(range(0, 24))
   print("Mean contact angle (°):", results.mean_angle)
   print("Std across batches (°):", results.std_angle)
   for batch in results.batches[:3]:
       print(
           f"Frames {batch.frames[0]}–{batch.frames[-1]}: "
           f"angle = {batch.angle:.2f}°, "
           f"R_eq = {batch.model_params['R_eq']:.2f} Å, "
           f"z_wall = {batch.model_params['zi_0']:.2f} Å"
       )

----

4. Output
---------

The returned :class:`CoupledFit2DResults` object exposes:

* ``mean_angle`` / ``std_angle`` — mean and std across batches.
* ``per_batch_angles`` — array of one angle per batch.
* ``batches`` — list of :class:`CoupledFit2DBatchResult` entries.
  Each batch carries ``angle``, ``model_params`` (the seven tanh
  parameters), and ``xi_grid`` / ``zi_grid`` / ``density`` — the
  binned density field used for the fit. Feed any batch (or the
  full results object) into :class:`DensityContourPlotter` to draw
  the density contour with the fitted spherical cap overlaid (see
  :doc:`visualization_evolution_density`).

Example printed output::

   Number of water molecules: 4000
   Mean contact angle (°): 99.11
   Std across batches (°): 0.0
   Frames 0–9: angle = 99.11°, R_eq = 42.13 Å, z_wall = 5.85 Å

----

5. Tips
-------

- **Grid bounds and cell width**: pick ``xi_f`` and ``zi_f`` so the
  droplet sits well inside the grid; pick ``dx`` and
  ``dz`` so each cell receives many atoms when pooling. As
  a rule of thumb, aim for at least 20 atoms per occupied cell after
  pooling across the batch. The range bounds are honoured exactly;
  the effective cell width is rounded so an integer number of cells
  fits, and may differ from the requested value by a few percent.
- **No ``grid_params``?** Leaving it ``None`` uses an atom-derived
  default: lateral half-box for ``xi``/``zi``, ``dx`` / ``dz`` = 0.5 Å
  (half the model's default interface thickness ``t1``). A warning
  is emitted to flag that the user didn't tune the grid.
- **Batch size**: the coupled fit benefits from statistics, so pool as
  many frames as your time-resolution needs allow. ``batch_size=-1``
  (the default) pools everything into one batch and returns a single
  angle.
- **Initial parameters**: the default initial guess
  ``[rho1, rho2, R_eq, z_c, z_0, t1, t2]`` is tuned for
  full-atomistic water at room temperature. Pass ``initial_params``
  explicitly if you see the fit's ``rho1`` or ``rho2`` pegged at the
  zero bound (the analyzer warns when this happens).
- **3D extension**: if you suspect significant deviation from
  axisymmetry, swap in :class:`CoupledFit3DAnalyzer`. Spherical
  droplets only; same API plus extra ``yi_*`` bin keys. The full
  tutorial is :doc:`coupled_fit_3d_tuto`.

For a side-by-side density contour plot with the fitted cap
overlaid, see :doc:`visualization_evolution_density`.

----

6. Alternative configurations
-----------------------------

6.1 Cylindrical droplet
^^^^^^^^^^^^^^^^^^^^^^^

The 2D coupled-fit analyzer handles cylindrical droplets out of
the box — pass ``droplet_geometry="cylinder_y"`` (or
``"cylinder_x"``). The projection switches from radial
(:math:`\xi = \sqrt{x^2 + y^2}`) to perpendicular-to-axis
(:math:`\xi = |x - x_c|`) and the density normalisation changes
from spherical (:math:`dV = 2\pi \xi\, d\xi\, dz`) to cartesian
(:math:`dV = L_y\, d\xi\, dz`, with :math:`L_y` the box length
along the cylinder axis):

.. code-block:: python

   analyzer = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(cylinder_fixture),
       atom_indices=oxygen_indices,
       droplet_geometry="cylinder_y",
       grid_params={
           "xi_0": 0.0,
           "xi_f": 100.0,
           "dx": 2.0,
           "zi_0": 0.0,
           "zi_f": 100.0,
           "dz": 4.0,
       },
   )

The seven-parameter model and the cap-angle formula are identical
to the spherical case; the geometry change is fully absorbed into
the projection and the volume normalisation.

6.2 Gaussian KDE density (smoother than the histogram)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default per-cell density is a top-hat histogram with
geometry-aware ``dV`` normalisation — fast, exact, and intrinsically
noisy at small per-cell atom counts. The seven-parameter tanh fit
will fail or converge to a degenerate minimum on per-frame batches
where the histogram density has too many empty cells.

Pass a :meth:`DensityEstimator.gaussian` instance to switch the
per-cell density to a 3D Gaussian KDE evaluated at the grid cell
centres — the same kernel ``rays`` (Gaussian) and ``grid`` (Gaussian)
use. The density field becomes smooth; per-cell Poisson noise
disappears at the cost of a small constant per-fit overhead:

.. code-block:: python

   from wetting_angle_kit.analysis import CoupledFit2DAnalyzer, DensityEstimator

   analyzer = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params=grid_params,
       density_estimator=DensityEstimator.gaussian(density_sigma=2.5),
       # batch_size=1 now becomes viable — the KDE density is smooth
       # enough that per-frame fits don't fall into the degenerate
       # ``t1`` minimum the histogram occasionally produces.
       temporal_aggregator=TemporalAggregator(batch_size=1),
   )

Pick the same ``density_sigma`` you would for ``rays`` (Gaussian) on
the same system (3 Å is the default; smaller for finer features,
larger for sparser systems). The recovered angle differs from the
binning variant by at most ~1° on well-pooled batches, but the
Gaussian variant is far more robust on small batches and on
systems with low atom density per cell.

6.3 Custom initial parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default initial guess
:math:`(\rho_1, \rho_2, R_{eq}, z_c, z_0, t_1, t_2) =
(10^{-3}, 0.03, 40, 20, 4, 1, 1)`
is tuned for full-atomistic water at room temperature in Å units.
If your simulation uses different units or a different liquid, pass
``initial_params=`` explicitly:

.. code-block:: python

   analyzer = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params=grid_params,
       initial_params=[1e-3, 0.02, 25.0, 8.0, 5.0, 1.0, 1.0],
   )

The analyzer emits a warning if any fitted parameter ends up
pinned at the physical lower bound (densities at 0, lengths at
:math:`10^{-6}`) — that's the usual sign your initial guess is
far from the true minimum or your grid bounds are wrong.

6.4 Single fully-pooled batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drop the ``temporal_aggregator`` argument (or set
``batch_size=-1``) to get one angle for the whole trajectory:

.. code-block:: python

   results = CoupledFit2DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params=grid_params,
   ).analyze(range(0, 24))
   print(results.batches[0].angle)  # single representative angle

This is the natural mode for the coupled fit — the NLLS
benefits from as much statistics as you can throw at it. Use
``batch_size=N`` only if you actually want time resolution
(e.g. to see contact-angle relaxation during a wetting event).
