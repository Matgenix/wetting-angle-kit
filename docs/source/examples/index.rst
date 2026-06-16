Examples
========

Ready-to-run example scripts demonstrating common workflows.

Parsing Trajectory Files
-------------------------

Parse different trajectory file formats (LAMMPS dump, ASE, XYZ) into a
unified ``(N, 3)`` coordinate array.

.. literalinclude:: ../../examples/parsing_trajectory_files.py
   :language: python
   :linenos:

----

Slicing-Pipeline Contact Angle
------------------------------

Per-frame angle via the composable :class:`TrajectoryAnalyzer` with the
ray-fan extractor and the slicing fitter.

.. literalinclude:: ../../examples/slicing_ca.py
   :language: python
   :linenos:

----

Whole-Fit Contact Angle with Bootstrap
--------------------------------------

Whole-shape sphere fit with the wall position taken from the actual
substrate atoms and a bootstrap uncertainty.

.. literalinclude:: ../../examples/whole_fit_ca.py
   :language: python
   :linenos:

----

Coupled-Fit Contact Angle
-------------------------

Coupled hyperbolic-tangent density-model fit via
:class:`CoupledFit2DAnalyzer` — one angle per pooled batch. The
example shows both density estimators (histogram default vs Gaussian
KDE).

.. literalinclude:: ../../examples/coupled_fit_ca.py
   :language: python
   :linenos:

----

Visualising a Per-Frame Droplet Snapshot
----------------------------------------

Pull a single slice's interface contour off a slicing-pipeline result
and render it with :class:`DropletSlicePlotter`.

.. literalinclude:: ../../examples/visualisation_slicing_traj.py
   :language: python
   :linenos:

----

Angle Evolution + Density Contour Plots
---------------------------------------

The two trajectory-level plotters
(:class:`AngleEvolutionPlotter` and :class:`DensityContourPlotter`)
on the same trajectory.

.. literalinclude:: ../../examples/visualisation_evolution_density.py
   :language: python
   :linenos:
