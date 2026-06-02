# wetting-angle-kit

[![tests](https://img.shields.io/github/actions/workflow/status/Matgenix/wetting-angle-kit/testing.yml?branch=main&label=tests)](https://github.com/Matgenix/wetting-angle-kit/actions/workflows/testing.yml)
[![code coverage](https://codecov.io/gh/Matgenix/wetting-angle-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/Matgenix/wetting-angle-kit)
[![pypi version](https://img.shields.io/pypi/v/wetting-angle-kit?color=blue)](https://pypi.org/project/wetting-angle-kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/wetting-angle-kit)](https://pypi.org/project/wetting-angle-kit/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-matgenix.github.io-blue)](https://matgenix.github.io/wetting-angle-kit)

wetting-angle-kit provides modular tools to parse MD trajectories (LAMMPS dump, XYZ, ASE) and compute droplet contact angles using two complementary approaches:

1. Slicing Method (per-frame circle fit) – robust against transient shape changes.
2. Binning Density Method – averages frames into a density field for a single representative angle.

The documentation is available [here](https://matgenix.github.io/wetting-angle-kit), you can find examples and tutorials.

## Installation

### Prerequisites

Before installing wetting-angle-kit, ensure you have the following prerequisites:

1. **Python 3.10 or higher**: Make sure you have Python 3.10 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

Core (only to analyse simple xyz trajectories):

```bash
pip install wetting-angle-kit
```

With OVITO:
```bash
pip install wetting-angle-kit[ovito]
```
With ASE:
```bash
pip install wetting-angle-kit[ase]
```
All optional:
```bash
pip install wetting-angle-kit[all]
```

#### Install OVITO

OVITO must be installed first in the conda environment and using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```

## Quick Start


```python
from wetting_angle_kit.analysis import (
    BinningTrajectoryAnalyzer,
    SlicingTrajectoryAnalyzer,
)
from wetting_angle_kit.parsers import XYZParser, XYZWaterFinder

trajectory_file = "trajectory.xyz"

# Identify water oxygen atoms by neighbor count. ``particle_type_wall``
# lists the symbols of the substrate atoms so they are excluded.
finder = XYZWaterFinder(trajectory_file, particle_type_wall=["C"])
oxygen_ids = finder.get_water_oxygen_indices(frame_index=0)

parser = XYZParser(trajectory_file)

slicing = SlicingTrajectoryAnalyzer(
    parser,
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
    delta_gamma=5,
)
results = slicing.analyze(frame_range=range(0, 50))
print(results.mean_angle, results.std_angle)

binning = BinningTrajectoryAnalyzer(
    parser,
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
)
results_binning = binning.analyze(frame_range=range(0, 200))
print(results_binning.mean_angle, results_binning.std_angle)
```
