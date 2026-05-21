Tutorial: Contact Angle Analysis (Sliced Method)
=================================================

This tutorial explains how to compute the contact angle of a droplet using the **sliced method** in ``wetting_angle_kit``.

----

1. Overview
-----------

The **sliced method** divides the droplet into slices (along the z-axis) and fits a geometric model (e.g. spherical) to the liquid–solid interface profile.
This is ideal for study the evolution of the angles among a trajectory.

----

2. Requirements
---------------

Before running the example, ensure you have installed:

.. code-block:: bash

   pip install wetting_angle_kit ase numpy

Example trajectory::

   tests/trajectories/traj_spherical_drop_4k.lammpstrj

----

3. Example Code
---------------

.. code-block:: python

   # Import necessary modules
   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.contact_angle_methods import contact_angle_analyzer

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
   # Using the 'sliced' method with a spherical model
   analyzer = contact_angle_analyzer(
       method="sliced",
       parser=parser,
       output_dir="result_dump_spherical_sliced",
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",  # Geometry fitting model
       delta_gamma=20,  # Azimuthal step (deg) for spherical slicing
   )

   # --- Step 6: Run the analysis ---
   results = analyzer.analyze([1])  # Analyze frame 1

   # --- Step 7: Display results ---
   print("Analysis results:", results)

----

4. Expected Output
------------------

After running the example, you'll see something like::

   Number of water molecules: 1320
   Analysis results: {
       'mean_angle': 94.46,
       'std_angle': 0.0,
       'angles': {1: 94.46},
       'frames_analyzed': [1],
       'method_metadata': {'frames_per_angle': 1},
   }

The ``analyze`` return dict has these keys:

* ``mean_angle`` — mean contact angle (°) across the analyzed frames.
* ``std_angle`` — standard deviation across frames.
* ``angles`` — mapping ``frame_index -> mean angle for that frame``.
* ``frames_analyzed`` — list of frame indices that were processed.
* ``method_metadata`` — method-specific info (e.g. number of frames per
  angle value).

Per-frame raw outputs (alfas, surfaces, popt) are saved as ``.npy`` files
inside the output directory.

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

**Example Script:** ``docs/examples/contact_angle_sliced/example_sliced.py``

.. code-block:: python

   """
   Example: Contact Angle Analysis Using the Sliced Method

   This example demonstrates how to perform a contact angle analysis
   using the 'sliced' method on a spherical droplet from a LAMMPS dump trajectory.
   """

   from wetting_angle_kit.parsers import LammpsDumpParser, LammpsDumpWaterFinder
   from wetting_angle_kit.contact_angle_methods import contact_angle_analyzer

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

   # --- Step 4: Create analyzer for the sliced method ---
   analyzer = contact_angle_analyzer(
       method="sliced",
       parser=parser,
       output_dir="result_dump_spherical_sliced",
       atom_indices=oxygen_indices,
       droplet_geometry="spherical",  # Fitting model
       delta_gamma=20,  # Azimuthal step (deg) for spherical slicing
   )

   # --- Step 5: Run analysis ---
   results = analyzer.analyze([1])  # Analyze frame 1

   # --- Step 6: Display results ---
   print("Analysis results:", results)
