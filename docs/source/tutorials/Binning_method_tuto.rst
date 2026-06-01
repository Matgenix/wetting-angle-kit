Tutorial: Contact Angle Analysis (Binning Method)
=================================================

This tutorial demonstrates how to compute the contact angle using the **binning method** in ``wetting_angle_kit``.
The method divides the simulation box into spatial bins to calculate the liquid–solid interface and the corresponding contact angle, for a group of frames.

----

1. Overview
-----------

The **binning method** works by:

1. Collecting the positions of water molecules (typically oxygen atoms).
2. Dividing the region of interest into bins in the **x–z** plane.
3. Computing density profiles and fitting the interface shape.
4. Deriving the contact angle from the interface curvature.

----

2. Prerequisites
----------------

Your trajectory file (e.g., a LAMMPS dump file) contain:

- Atom IDs, types, and positions
- Liquid particles (in this cas Water molecules: O and H atoms)

Example trajectory::

   tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj

----

3. Example Script
-----------------

.. code-block:: python

   # Import necessary modules
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.analysis import BinningContactAngleAnalyzer

   # --- Step 1: Define the trajectory file ---
   filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

   # --- Step 2: Initialize the water molecule finder ---
   # This identifies O and H atoms in water molecules
   wat_find = LammpsDumpWaterFinder(
       filename,
       particle_type_wall={3},  # Wall atom types
       oxygen_type=1,  # Oxygen atom type
       hydrogen_type=2,  # Hydrogen atom type
   )

   # --- Step 3: Get oxygen atom indices for the first frame ---
   oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
   print("Number of water molecules:", len(oxygen_indices))

   # --- Step 4: Define binning parameters ---
   binning_params = {
       "xi_0": 0.0,  # Minimum x-coordinate
       "xi_f": 100.0,  # Maximum x-coordinate
       "nbins_xi": 50,  # Number of bins along x
       "zi_0": 0.0,  # Minimum z-coordinate
       "zi_f": 100.0,  # Maximum z-coordinate
       "nbins_zi": 25,  # Number of bins along z
   }

   # --- Step 5: Initialize the parser ---
   parser = LammpsDumpParser(filename)

   # --- Step 6: Create the contact angle analyzer ---
   analyzer = BinningContactAngleAnalyzer(
       parser=parser,
       atom_indices=oxygen_indices,
       droplet_geometry="cylinder_y",  # Interface fitting model
       width_cylinder=21,  # Width parameter for interface fit
       binning_params=binning_params,
   )

   # --- Step 7: Run analysis for a frame range ---
   results = analyzer.analyze([1])  # Analyze frame 1
   print("Mean contact angle (°):", results.mean_angle)

----

4. Output
---------

Running this example will:

- Parse the trajectory.
- Compute the interface shape and local contact angle for each batch.
- Return a :class:`BinningResults` dataclass holding angles, density fields
  and fitted isolines for every batch (no files written).

Example printed output::

   Number of water molecules: 4000
   Mean contact angle (°): 94.58987060394456

The returned ``results`` object exposes ``mean_angle``, ``std_angle``,
``angles_per_batch`` and a ``batches`` list whose entries carry the
density field (``xi_cc``, ``zi_cc``, ``rho_cc``) and the fitted
droplet / wall isoline coordinates. Feed it directly to
:class:`BinningTrajectoryPlotter` to draw the interactive density
contour with the fitted semi-circle:

.. code-block:: python

   from wetting_angle_kit.visualization import BinningTrajectoryPlotter

   plotter = BinningTrajectoryPlotter(results)
   fig = plotter.plot_density_contour(batch_index=0)
   fig.show()

----

5. Tips
-------

- Adjust ``xi_f``, ``zi_f``, and the bin counts (``nbins_xi``, ``nbins_zi``) according to your simulation box dimensions.
- If the wall surface is not flat or the system is tilted, pre-align it before analysis.
- Multi-batch analysis: ``analyzer.analyze(range(0, 100), split_factor=10)`` splits the frame range into batches of ten frames each.
