# wetting_angle_kit

[![Tests](https://github.com/Matgenix/wetting_angle_kit/actions/workflows/testing.yml/badge.svg)](https://github.com/Matgenix/wetting_angle_kit/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/Matgenix/wetting_angle_kit/branch/main/graph/badge.svg)](https://codecov.io/gh/Matgenix/wetting_angle_kit)
[![PyPI version](https://img.shields.io/pypi/v/wetting_angle_kit.svg)](https://pypi.org/project/wetting_angle_kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/wetting_angle_kit.svg)](https://pypi.org/project/wetting_angle_kit/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-matgenix.github.io-blue)](https://matgenix.github.io/wetting-angle-kit)

wetting_angle_kit provides modular tools to parse MD trajectories (LAMMPS dump, XYZ, ASE) and compute droplet contact angles using two complementary approaches:

1. Sliced Method (per-frame circle fit) – robust against transient shape changes.
2. Binning Density Method – averages frames into a density field for a single representative angle.

The documentation is available [here](https://matgenix.github.io/wetting-angle-kit), you can find examples and tutorials.

## Installation

### Prerequisites

Before installing wetting_angle_kit, ensure you have the following prerequisites:

1. **Python 3.10 or higher**: Make sure you have Python 3.10 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

Core (only to analyse simple xyz trajectories):

```bash
pip install wetting_angle_kit
```

With OVITO:
```bash
pip install wetting_angle_kit[ovito]
```
With ASE:
```bash
pip install wetting_angle_kit[ase]
```
All optional:
```bash
pip install wetting_angle_kit[all]
```

#### Install OVITO

OVITO must be installed first in the conda environment and using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```

## Quick Start


```python
from wetting_angle_kit.contact_angle_methods import (
    BinningContactAngleAnalyzer,
    SlicedContactAngleAnalyzer,
)
from wetting_angle_kit.parsers import XYZParser, XYZWaterFinder

trajectory_file = "trajectory.xyz"

# Identify water oxygen atoms by neighbor count. ``particle_type_wall``
# lists the symbols of the substrate atoms so they are excluded.
finder = XYZWaterFinder(trajectory_file, particle_type_wall=["C"])
oxygen_ids = finder.get_water_oxygen_indices(frame_index=0)

parser = XYZParser(trajectory_file)

sliced = SlicedContactAngleAnalyzer(
    parser,
    output_dir="out_sliced",
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
    delta_gamma=5,
)
results = sliced.analyze(frame_range=range(0, 50))
print(results["mean_angle"], results["std_angle"])

binning = BinningContactAngleAnalyzer(
    parser,
    output_dir="out_binned",
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
)
results_binning = binning.analyze(frame_range=range(0, 200))
print(results_binning["mean_angle"], results_binning["std_angle"])
```
