Tutorial: Comparing Trajectory Analysis Methods
================================================

This tutorial demonstrates how to use the ``BinningTrajectoryAnalyzer`` and ``SlicedTrajectoryAnalyzer`` classes to analyze and compare contact angle and surface area data from trajectory simulations.

----

Introduction
------------

The ``BinningTrajectoryAnalyzer`` and ``SlicedTrajectoryAnalyzer`` classes are designed to analyze trajectory data, specifically focusing on **surface area** and **contact angle** statistics. These tools are useful for comparing different analysis methods and visualizing results.

----

Setup and Initialization
-------------------------

Import the Classes
^^^^^^^^^^^^^^^^^^

Ensure you have the required classes imported:

.. code-block:: python

   from wetting_angle_kit.visualization_angles.binning_analyzer import (
       BinningTrajectoryAnalyzer,
   )
   from wetting_angle_kit.visualization_angles.sliced_analyzer import (
       SlicedTrajectoryAnalyzer,
   )
   from wetting_angle_kit.visualization_angles.comparison_methods import (
       MethodComparison,
   )

Initialize the Analyzers
^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the directories containing your trajectory data:

.. code-block:: python

   directories = [
       "sliced_analysis_CA/result_dump_traj_2k_reduce_binned",
       "sliced_analysis_CA/result_dump_traj_500_reduce_binned",
       "sliced_analysis_CA/result_dump_traj_1k_reduce_binned",
       "sliced_analysis_CA/result_dump_traj_8k_reduce_binned",
   ]

   # Initialize the analyzers
   sliced = SlicedTrajectoryAnalyzer(directories)
   binning = BinningTrajectoryAnalyzer(directories)

----

Running the Analysis
--------------------

Analyze Data
^^^^^^^^^^^^

Run the analysis for both methods:

.. code-block:: python

   sliced.analyze()
   binning.analyze()

Example Output
^^^^^^^^^^^^^^

::

   Directory: sliced_analysis_CA/result_dump_traj_2k_reduce_binned
     Method: Sliced Analysis
     Mean Surface Area: 2770.0659
     Mean Contact Angle: 91.7015°

   Directory: binning_analysis_CA/result_dump_traj_2k_reduce_binned
     Method: Binning Analysis
     Mean Surface Area: 2748.5427
     Mean Contact Angle: 91.9236°

----

Interpreting the Output
------------------------

- **Mean Surface Area**: The average surface area for each trajectory.
- **Mean Contact Angle**: The average contact angle for each trajectory.
- **Standard Deviation**: Indicates the variability of the data.

----

Visualisation
-------------

Plot Mean Angle vs Surface Area
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sliced.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface_sliced.png")
   binning.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface_binning.png")

Plot Median Angle Evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the sliced method, plot the evolution of median angles:

.. code-block:: python

   sliced.plot_median_alfas_evolution(save_path="evolution_of_angles_sliced_method.png")

----

Method Comparison
-----------------

Compare Statistics
^^^^^^^^^^^^^^^^^^

Use the ``MethodComparison`` class to compare the two methods:

.. code-block:: python

   comparison = MethodComparison([sliced, binning])
   comparison.plot_side_by_side_comparison(save_path="comparison.png")
   comparison.compare_statistics()

Example Output
^^^^^^^^^^^^^^

::

   ======================================================================
   METHOD COMPARISON STATISTICS
   ======================================================================
   Sliced Analysis:
   ----------------------------------------------------------------------
     sliced_analysis_CA/traj_2k/:
       Mean Surface Area: 2770.0659 ± 15.2001
       Mean Angle: 91.7015° ± 5.6130°
     Overall Statistics:
       Total samples: 196
       Mean Surface Area: 4001.0215
       Mean Angle: 91.8326°
       Std Angle: 6.2027°

   Binning Analysis:
   ----------------------------------------------------------------------
     binning_analysis_CA/traj_2k:
       Mean Surface Area: 2748.5427 ± 0.0000
       Mean Angle: 91.9236° ± 0.0000°
     Overall Statistics:
       Total samples: 4
       Mean Surface Area: 4022.1019
       Mean Angle: 92.0876°
       Std Angle: 0.2391°

----

Conclusion
----------

- The ``SlicedTrajectoryAnalyzer`` provides more detailed statistics with higher sample counts.
- The ``BinningTrajectoryAnalyzer`` offers a simplified, binning approach.
- Use the comparison tools to visualize and interpret differences between methods.

Additional Notes
----------------

- Ensure your data directories are correctly formatted and contain the required log files.
