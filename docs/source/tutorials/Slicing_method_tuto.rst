Tutorial: Contact Angle Analysis (Slicing Method)
=================================================

This tutorial explains how to compute the contact angle of a droplet using the **slicing method** in ``wetting_angle_kit``.

----

1. Overview
-----------

The **slicing method** divides the droplet into slices (along the z-axis) and fits a geometric model (e.g. spherical) to the liquid–solid interface profile.
This is ideal for study the evolution of the angles among a trajectory.

----

2. Requirements
---------------

Before running the example, ensure you have installed:

.. code-block:: bash

   pip install wetting-angle-kit ase numpy

Example trajectory::

   tests/trajectories/traj_spherical_drop_4k.lammpstrj

----

3. Example Code
---------------

.. code-block:: python

   # Import necessary modules
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.analysis import SlicingContactAngleAnalyzer

   # --- Step 1: Define the trajectory file ---
   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

   # --- Step 2: Initialize the water molecule finder ---
   wat_find = LammpsDumpWaterFinder(
       filename,
       particle_type_wall={3},  # Wall particle types
       oxygen_type=1,  # Oxygen atom type
       hydrogen_type=2,
   )  # Hydrogen atom type

   # --- Step 3: Identify oxygen atom indices ---
   oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
   print("Number of water molecules:", len(oxygen_indices))

   # --- Step 4: Initialize the parser ---
   parser = LammpsDumpParser(filename)

   # --- Step 5: Create the contact angle analyzer ---
   # Using the slicing method with a spherical model
   analyzer = SlicingContactAngleAnalyzer(
       parser=parser,
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",  # Geometry fitting model
       delta_gamma=20,  # Azimuthal step (deg) for spherical slicing
   )

   # --- Step 6: Run the analysis ---
   results = analyzer.analyze([1])  # Analyze frame 1

   # --- Step 7: Display results ---
   print("Mean contact angle (°):", results.mean_angle)
   print("Std contact angle (°):", results.std_angle)
   print("Frames analyzed:", results.frames)

----

4. Expected Output
------------------

After running the example, you'll see something like::

   Number of water molecules: 1320
   Mean contact angle (°): 94.46
   Std contact angle (°): 0.0
   Frames analyzed: [1]

``analyze`` returns a :class:`SlicingResults` dataclass with the
following convenience attributes:

* ``mean_angle`` — mean contact angle (°) across the analyzed frames.
* ``std_angle`` — standard deviation across frames.
* ``per_frame_mean_angles`` — array of per-frame mean angles (one per slice
  aggregated to a single number).
* ``frames`` — list of frame indices that were processed.
* ``angles`` / ``surfaces`` / ``popts`` — raw per-frame data passed
  directly to :class:`SlicingTrajectoryPlotter` for visualization.
* ``method_metadata`` — method-specific info (e.g. number of frames per
  angle value).

----

5. Tips
-------

- Use ``droplet_geometry='spherical'`` for droplets and ``droplet_geometry='cylinder_y'`` for cylindrical droplet on the y axis or ``'cylinder_x'`` for cylinder on the x axis.
- Adjust ``delta_gamma`` for the spherical mode (azimuthal step in
  degrees between successive slices — smaller = more slices, more
  detail, more cost). For very small droplets, also raise
  ``points_per_angstrom`` (default 1.0) on the analyzer to densify the
  per-ray sampling used by the interface fit.
- To analyze multiple frames:

.. code-block:: python

   results = analyzer.analyze(range(0, 50, 10))

- Output files include raw interface data and optional plots (if enabled).

----

6. Related Files
----------------

**Example Script:** ``docs/examples/contact_angle_slicing/example_slicing.py``

.. code-block:: python

   """
   Example: Contact Angle Analysis Using the Slicing Method

   This example demonstrates how to perform a contact angle analysis
   using the 'slicing' method on a spherical droplet from a LAMMPS dump trajectory.
   """

   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.analysis import SlicingContactAngleAnalyzer

   # --- Step 1: Define input trajectory ---
   filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

   # --- Step 2: Identify water molecules ---
   wat_find = LammpsDumpWaterFinder(
       filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2  # Wall atom types
   )

   oxygen_indices = wat_find.get_water_oxygen_ids(frame_index=0)
   print(f"Number of water molecules: {len(oxygen_indices)}")

   # --- Step 3: Initialize parser ---
   parser = LammpsDumpParser(filename)

   # --- Step 4: Create analyzer for the slicing method ---
   analyzer = SlicingContactAngleAnalyzer(
       parser=parser,
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",  # Fitting model
       delta_gamma=20,  # Azimuthal step (deg) for spherical slicing
   )

   # --- Step 5: Run analysis ---
   results = analyzer.analyze([1])  # Analyze frame 1

   # --- Step 6: Display results ---
   print("Mean contact angle (°):", results.mean_angle)
   print("Std contact angle (°):", results.std_angle)
