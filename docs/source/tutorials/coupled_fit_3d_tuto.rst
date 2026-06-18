Tutorial: 3D Coupled-Fit Analyzer
=================================

:class:`CoupledFit3DAnalyzer` is the 3D extension of
:class:`CoupledFit2DAnalyzer`. Instead of projecting atoms to a
2D ``(xi, zi)`` plane and exploiting radial symmetry, it builds the
full 3D density ``rho(xi, yi, zi)`` and fits a nine-parameter
hyperbolic-tangent density model directly:

.. math::

   \rho(\xi, \eta, z) \;=\; g(r) \cdot h(z - z_0),
   \qquad r = \sqrt{(\xi - \xi_c)^2 + (\eta - \eta_c)^2 + (z - z_c)^2},

with two extra horizontal-centre parameters
:math:`\xi_c, \eta_c` over the 2D model. See
:doc:`../introduction/theoretical_foundations` section 6 for the full
model.

----

1. When to pick the 3D variant?
-------------------------------

The 2D analyzer assumes axisymmetry: the coupled fit collapses the
droplet onto a 2D ``(xi, zi)`` profile via the radial coordinate
:math:`\xi = \sqrt{x^2 + y^2}`. That assumption is excellent for
clean spherical droplets but breaks if the droplet is asymmetric —
e.g. on a heterogeneous wall, near a step edge, or under an external
field.

The 3D analyzer **does not** assume axisymmetry. It still fits a
spherical cap (the radial sigmoid in the model is a sphere), but the
two extra parameters :math:`\xi_c, \eta_c` let the fit identify the
horizontal cap centre instead of requiring it to coincide with the
COM. Useful when:

* you suspect the droplet's footprint is shifted away from the
  geometric centre of mass;
* you want to verify visually (via
  :class:`DensityContourPlotter`) that the recovered radial profile
  is consistent in different azimuthal directions.

For purely axisymmetric droplets, the 2D analyzer is several times
cheaper for the same statistical quality. The 3D analyzer is the
right choice when you suspect asymmetry, or when you want the
azimuthally-collapsed density plot from a 3D fit as cross-validation.

Cylindrical droplets are **rejected at construction**: their
translational symmetry along the cylinder axis already collapses the
3D problem onto the 2D one, so the 3D analyzer would just be
wasting work.

----

2. Worked example
-----------------

.. code-block:: python

   from wetting_angle_kit.analysis import CoupledFit3DAnalyzer
   from wetting_angle_kit.analysis.temporal import TemporalAggregator
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"
   oxygen_indices = LammpsDumpWaterFinder(
       filename, oxygen_type=1, hydrogen_type=2
   ).get_water_oxygen_indices(frame_index=0)

   # 3D grid spec. xi/yi are in the droplet-centred frame; zi is in the
   # lab frame so the wall position retains physical meaning.
   grid_params = {
       "xi_0": -30.0,
       "xi_f": 30.0,
       "dx": 3.2,
       "yi_0": -30.0,
       "yi_f": 30.0,
       "dy": 3.2,
       "zi_0": 0.0,
       "zi_f": 60.0,
       "dz": 4.0,
   }

   analyzer = CoupledFit3DAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       grid_params=grid_params,
       # 3D density grids need more frames per batch than 2D ones to
       # reach the same per-cell noise; default is fully pooled.
       temporal_aggregator=TemporalAggregator(batch_size=-1),
   )
   results = analyzer.analyze(range(0, 100))
   batch = results.batches[0]

   print(f"Angle: {batch.angle:.2f}°")
   print(f"R_eq:  {batch.model_params['R_eq']:.2f} Å")
   print(
       f"Cap centre (x_c, y_c, z_c): "
       f"({batch.model_params['xi_c']:.2f}, "
       f"{batch.model_params['yi_c']:.2f}, "
       f"{batch.model_params['zi_c']:.2f}) Å"
   )
   print(f"Wall z_0: {batch.model_params['zi_0']:.2f} Å")

----

3. Reading the recovered centre
-------------------------------

For an axisymmetric droplet the recovered :math:`(\xi_c, \eta_c)`
should collapse to ``(0, 0)`` — the COM-centred grid is set up
exactly so that's what zero offset means. Substantial non-zero
values are diagnostic:

* :math:`|\xi_c| + |\eta_c| < 0.5` Å: cleanly axisymmetric.
* a few Å: mild asymmetry — the cap is genuinely off-axis, or the
  per-frame PBC recentering disagrees with the cap centre.
* tens of Å: something is wrong — the grid bounds are too tight, the
  initial parameters are off, or the droplet is split across a
  periodic boundary the PBC recentering didn't catch.

The horizontal-centre recovery is the main reason to prefer the 3D
analyzer over the 2D one when you suspect a non-spherical-cap
geometry: it tells you *whether* the assumption holds, not just the
angle conditional on it holding.

----

4. Visualising the 3D density
-----------------------------

The 3D density tensor is too rich to plot directly. The
:class:`DensityContourPlotter` collapses the 3D density azimuthally
onto a 2D ``(r, z)`` plane (binning by :math:`r =
\sqrt{x^2 + y^2}` and averaging per :math:`z`-slice), then renders
it exactly like the 2D variant — with the fitted spherical cap
overlaid:

.. code-block:: python

   from wetting_angle_kit.visualization import DensityContourPlotter

   # Single batch — azimuthally averaged onto (r, z).
   fig = DensityContourPlotter(batch, label="spherical_4k").plot()
   fig.show()

   # Whole results object — averaged across batches first, then
   # azimuthally averaged.
   fig = DensityContourPlotter(results, label="spherical_4k").plot()
   fig.show()

The default title indicates the azimuthal collapse so the plot is
unambiguous.

----

5. Cross-check: 2D vs 3D
------------------------

On an axisymmetric droplet, the 2D and 3D analyzers should recover
the same angle within a few degrees. It's a useful sanity check:

.. code-block:: python

   from wetting_angle_kit.analysis import (
       CoupledFit2DAnalyzer,
       CoupledFit3DAnalyzer,
   )

   # Same trajectory, same frames; pick comparable grids.
   grid_2d = {
       "xi_0": 0.0,
       "xi_f": 30.0,
       "dx": 1.0,
       "zi_0": 0.0,
       "zi_f": 60.0,
       "dz": 1.0,
   }
   grid_3d = {
       "xi_0": -30.0,
       "xi_f": 30.0,
       "dx": 3.2,
       "yi_0": -30.0,
       "yi_f": 30.0,
       "dy": 3.2,
       "zi_0": 0.0,
       "zi_f": 60.0,
       "dz": 4.0,
   }

   a2d = (
       CoupledFit2DAnalyzer(
           parser=LammpsDumpParser(filename),
           atom_indices=oxygen_indices,
           droplet_geometry="spherical",
           grid_params=grid_2d,
       )
       .analyze([1])
       .batches[0]
   )
   a3d = (
       CoupledFit3DAnalyzer(
           parser=LammpsDumpParser(filename),
           atom_indices=oxygen_indices,
           droplet_geometry="spherical",
           grid_params=grid_3d,
       )
       .analyze([1])
       .batches[0]
   )

   print(f"2D: {a2d.angle:.2f}°")
   print(f"3D: {a3d.angle:.2f}°")
   print(f"|Δ|: {abs(a2d.angle - a3d.angle):.2f}°")

On the test fixture this gives 94.46° (2D) vs 95.42° (3D), drift
≈ 0.96° — within the expected noise from the sparser 3D grid.

----

6. Tips
-------

- **Grid resolution**: 3D grids need substantially more atoms per
  cell than 2D ones for comparable noise. Pool more frames per
  batch (``batch_size`` larger), or use coarser grids (~25 bins per
  axis is usually enough). The 9-parameter fit doesn't benefit from
  arbitrarily fine grids the way one might naively expect.
- **Initial parameters**: the default initial guess
  ``[rho1, rho2, R_eq, xi_c, yi_c, zi_c, z_0, t1, t2]`` is tuned
  for full-atomistic water at room temperature with ``xi_c =
  yi_c = 0``. Override via ``initial_params`` if you have a strong
  prior on the cap geometry.
- **No cylinder support**: pass ``droplet_geometry="cylinder_y"``
  to ``CoupledFit3DAnalyzer`` and you'll get a ``ValueError``
  at construction explaining the design choice; route cylindrical
  droplets through :class:`CoupledFit2DAnalyzer` instead.
