# Tutorial: Comparing Trajectory Analysis Methods

This tutorial demonstrates how to use the `BinningTrajectoryAnalyzer` and `SlicingTrajectoryAnalyzer` classes to analyze and compare contact angle and surface area data from trajectory simulations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Initialization](#setup-and-initialization)
3. [Running the Analysis](#running-the-analysis)
4. [Interpreting the Output](#interpreting-the-output)
5. [Visualization](#visualization)
6. [Method Comparison](#method-comparison)
7. [Conclusion](#conclusion)

---

## Introduction
The `BinningTrajectoryAnalyzer` and `SlicingTrajectoryAnalyzer` classes are designed to analyze trajectory data, specifically focusing on **surface area** and **contact angle** statistics. These tools are useful for comparing different analysis methods and visualizing results.

---

## Setup and Initialization

### Import the Classes
Ensure you have the required classes imported:

```python
from wetting_angle_kit.visualization import (
    BinningTrajectoryAnalyzer,
    MethodComparison,
    SlicingTrajectoryAnalyzer,
)
```
---

## Initialize the Analyzers
Specify the directories containing your trajectory data:

```python
directories = [
    "slicing_analysis_CA/result_dump_traj_500_binned",
    "slicing_analysis_CA/result_dump_traj_1k_binned",
    "slicing_analysis_CA/result_dump_traj_2k_binned",
    "slicing_analysis_CA/result_dump_traj_4k_binned",
]

# Initialize the analyzers
slicing = SlicingTrajectoryAnalyzer(directories)
binning = BinningTrajectoryAnalyzer(directories)
```
---
## Running the Analysis

### Analyze Data

Run the analysis for both methods:

```python
slicing.analyze()
binning.analyze()
```

### Example Output:

```text
Directory: slicing_analysis_CA/result_dump_traj_2k_reduce_binned
  Method: Slicing Analysis
  Mean Surface Area: 2770.0659
  Mean Contact Angle: 91.7015°

Directory: binning_analysis_CA/result_dump_traj_2k_reduce_binned
  Method: Binning Analysis
  Mean Surface Area: 2748.5427
  Mean Contact Angle: 91.9236°
```
---
### Interpreting the Output

- Mean Surface Area: The average surface area for each trajectory.
- Mean Contact Angle: The average contact angle for each trajectory.
- Standard Deviation: Indicates the variability of the data.
---
## Visualisation

### Plot Mean Angle vs Surface Area

```python
slicing.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface_slicing.png")
binning.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface_binning.png")
```
### Plot Median Angle Evolution
For the slicing method, plot the evolution of median angles:
```python
slicing.plot_median_alfas_evolution(save_path="evolution_of_angles_slicing_method.png")
```
## Method Comparison

### Compare Statistics
Use the MethodComparison class to compare the two methods:

```python
comparison = MethodComparison([slicing, binning])
comparison.plot_side_by_side_comparison(save_path="comparison.png")
print(comparison.compare_statistics())
```
### Example Output:

```text
======================================================================
METHOD COMPARISON STATISTICS
======================================================================
Slicing Analysis:
----------------------------------------------------------------------
  slicing_analysis_CA/traj_2k/:
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
```

## Conclusion
- The SlicingTrajectoryAnalyzer provides more detailed statistics with higher sample counts.

- The BinningTrajectoryAnalyzer offers a simplified, binning approach.

- Use the comparison tools to visualize and interpret differences between methods.

## Additional Notes
- Ensure your data directories are correctly formatted and contain the required log files.
