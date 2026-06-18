Tutorial: Whole-Shape Fit with Bootstrap Uncertainty
====================================================

This tutorial covers the **whole-fit pipeline**: an algebraic
sphere or cylinder fit to the entire interface shell at once, with
optional bootstrap resampling for an angle uncertainty. Pair it with
:class:`WallDetector.explicit` or :class:`WallDetector.from_atoms`
when you have a known wall position from the simulation setup.

----

1. Overview
-----------

The whole-fit pipeline differs from the slicing pipeline in two
places:

1. The :class:`InterfaceExtractor` returns a single ``(N, 3)`` shell
   array — every ray's interface point pooled into one cloud rather
   than divided into per-slice 2D sub-clouds.
2. The :class:`SurfaceFitter.whole()` runs **one** algebraic Taubin fit
   on that shell — sphere fit for spherical droplets, cylinder fit
   (algebraic circle in ``(x, z)`` with translational invariance
   along ``y``) for cylindrical droplets. The contact angle follows
   from the cap geometry ``cos θ = (z_wall - z_center) / R``.

If ``bootstrap_samples > 0`` the fit is repeated on that many
bootstrap resamples of the shell, and the resulting standard
deviation of the angles is reported as
:attr:`WholeBatchResult.angle_std`.

----

2. Wall detector pairing
------------------------

The full-sphere Fibonacci ray fan
(:meth:`SpaceSampling.rays` with ``n_rays_sphere=...``)
emits rays from the droplet COM in all directions, including
downward. Those downward rays hit the wall plane and contribute
interface points right at the wall, so the lowest shell point lands
at ``z ≈ z_wall``. As a result, :meth:`WallDetector.min_plus_offset`
*does* work with this pipeline.

The two physical alternatives are usually more reliable:

* :meth:`WallDetector.from_atoms` reads wall atom positions from the
  trajectory and places the wall plane at the mean of the top atomic
  layer. Best when the substrate is explicitly modelled.
* :meth:`WallDetector.explicit` lets you supply ``z_wall`` directly
  — handy when the wall position is known a priori from the
  simulation setup.

The slicing-mode equivalent of this consideration is much less
sensitive because the wall enters only as a horizontal line in each
slice's 2D fit, but for the whole-shape fit the recovered angle
depends quite linearly on the wall z, so it pays to be honest about
the wall position.

----

3. Example Code
---------------

.. code-block:: python

   from wetting_angle_kit.analysis import (
       DensityEstimator,
       InterfaceExtractor,
       SpaceSampling,
       SurfaceFitter,
       TrajectoryAnalyzer,
       WallDetector,
   )
   from wetting_angle_kit.parsers import (
       LammpsDumpParser,
       LammpsDumpWallParser,
       LammpsDumpWaterFinder,
   )

   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

   # --- Step 1: Identify water-oxygen and wall-atom indices ---
   wat_find = LammpsDumpWaterFinder(filename, oxygen_type=1, hydrogen_type=2)
   oxygen_indices = wat_find.get_water_oxygen_indices(frame_index=0)

   # Wall parser: ``liquid_particle_types`` lists what to EXCLUDE
   # (i.e. the liquid), leaving the wall atoms.
   wall_parser = LammpsDumpWallParser(filename, liquid_particle_types=[1, 2])
   carbon_indices = wall_parser.parse(frame_index=0)  # uses internal IDs

   # --- Step 2: Build the whole-fit analyzer ---
   # Strategies: full-sphere Fibonacci ray fan + whole-fit + from_atoms wall.
   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(filename),
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",
       interface_extractor=InterfaceExtractor(
           sampling=SpaceSampling.rays(
               n_rays_sphere=400,  # 400 rays uniformly over the full sphere
           ),
           density=DensityEstimator.gaussian(density_sigma=3.0),
       ),
       surface_fitter=SurfaceFitter.whole(
           surface_filter_offset=3.0,
           bootstrap_samples=100,  # 100 bootstrap resamples → angle_std
       ),
       wall_detector=WallDetector.min_plus_offset(),
       wall_atom_indices=carbon_indices,  # routed to the wall detector
   )

   # --- Step 3: Run the analysis ---
   results = analyzer.analyze([1])
   batch = results.batches[0]
   print(f"angle = {batch.angle:.2f}°  ± {batch.angle_std:.2f}° (bootstrap)")
   print(f"R     = {batch.popt[3]:.2f} Å, z_wall = {batch.z_wall:.2f} Å")
   print(f"rms residual on the shell = {batch.rms_residual:.2f} Å")

----

4. Expected Output
------------------

On the water/graphene fixture::

   angle = 99.13° ± 0.41° (bootstrap)
   R     = 37.54 Å, z_wall = 4.90 Å
   rms residual on the shell = 1.27 Å

The recovered ``R`` is *not* the physical droplet radius — it's the
sphere that best fits the recovered shell, which sits slightly inside
the actual interface because of the Gaussian-KDE smoothing. The
contact angle reading is unaffected by this; the cap geometry is
self-consistent with whichever ``R`` and ``z_c`` the fit produces.

----

5. Comparing whole vs slicing
-----------------------------

Both the whole and slicing pipelines on this fixture recover ~95°
when paired with :meth:`WallDetector.min_plus_offset(0)` (the
"natural" interface-derived baseline). When you switch the
wall detector to ``from_atoms`` or ``explicit`` at the actual
graphene top (≈ 4.9 Å), both pipelines move up to ~99°. The two
methods are physically consistent — the slicing pipeline gives you
an inter-slice ``σ`` for free; the whole pipeline gives you a
bootstrap ``σ`` instead.

The choice between them is mostly about what kind of uncertainty you
want to report:

* per-slice scatter (slicing): tells you whether the droplet is
  symmetric across the chosen slice axis.
* bootstrap σ (whole): tells you how well-determined the fit is given
  the shell points; doesn't expose asymmetry.

----

6. Tips
-------

- **Number of rays** (``n_rays_sphere``): 400 is plenty for a
  ~30 Å droplet; you'd push to 800–1600 for larger droplets where
  the shell point density per square Å matters.
- **Bootstrap samples**: 100 is enough to get an ``angle_std``
  reliable to two significant figures; 1000 will tighten that but
  costs ~10× more.
- **Cylinder droplets** still work — pair the whole fitter with
  :meth:`SpaceSampling.rays` configured with
  ``delta_cylinder`` and ``delta_polar`` instead of ``n_rays_sphere``.
  The fitter automatically does a 2D circle fit per the cylinder
  axis convention.

----

7. Alternative configurations
-----------------------------

7.1 Cylindrical whole-fit
^^^^^^^^^^^^^^^^^^^^^^^^^

For a cylindrical droplet, the whole fitter exploits translational
symmetry along the cylinder axis: it does **one** 2D circle fit in
the ``(x, z)`` plane to the entire shell, treating the
:math:`y`-coordinate as ignorable. The same recipe, with the
cylinder-mode extractor parameters:

.. code-block:: python

   analyzer = TrajectoryAnalyzer(
       parser=LammpsDumpParser(cylinder_fixture),
       atom_indices=oxygen_indices,
       droplet_geometry="cylinder_y",  # or "cylinder_x"
       interface_extractor=InterfaceExtractor(
           sampling=SpaceSampling.rays(delta_cylinder=5.0, delta_polar=8.0),
           density=DensityEstimator.gaussian(density_sigma=3.0),
       ),
       surface_fitter=SurfaceFitter.whole(
           surface_filter_offset=3.0,
           bootstrap_samples=100,
       ),
       wall_detector=WallDetector.min_plus_offset(offset=0.0),
   )

The recovered ``popt`` for the cylindrical case is a 4-element
array ``[xc, zc, R, z_wall]`` rather than the 5-element spherical
``[xc, yc, zc, R, z_wall]`` — the cylinder axis :math:`y` doesn't
participate in the fit.

7.2 Explicit wall position
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the wall plane is known a priori from the simulation setup
(e.g. a 9-3 LJ wall at a specific :math:`z`-coordinate, or you
simply read the top atomic layer's :math:`z` in an external script):

.. code-block:: python

   wall_detector = WallDetector.explicit(z_wall=5.0)

No ``wall_atom_indices`` argument needed on the analyzer in this
case — the explicit detector ignores any wall-atom data. Useful
both for whole-fit and slicing pipelines.

7.3 ``rays`` (binning) alternative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same Fibonacci-sphere geometry, but the density along each ray is
estimated with a 1D top-hat histogram instead of a Gaussian KDE:

.. code-block:: python

   interface_extractor = InterfaceExtractor(
       sampling=SpaceSampling.rays(n_rays_sphere=400),
       density=DensityEstimator.binning(bin_width=3.0),
   )
