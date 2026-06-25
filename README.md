# wetting-angle-kit

[![tests](https://img.shields.io/github/actions/workflow/status/Matgenix/wetting-angle-kit/testing.yml?branch=main&label=tests)](https://github.com/Matgenix/wetting-angle-kit/actions/workflows/testing.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/Matgenix/wetting-angle-kit/deploy-docs.yml?branch=main&label=docs)](https://github.com/Matgenix/wetting-angle-kit/actions/workflows/deploy-docs.yml)
[![code coverage](https://codecov.io/gh/Matgenix/wetting-angle-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/Matgenix/wetting-angle-kit)
[![pypi version](https://img.shields.io/pypi/v/wetting-angle-kit?color=blue)](https://pypi.org/project/wetting-angle-kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/wetting-angle-kit)](https://pypi.org/project/wetting-angle-kit/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-matgenix.github.io-blue)](https://matgenix.github.io/wetting-angle-kit)

**wetting-angle-kit** is a Python library for measuring contact angles directly from molecular dynamics trajectories.

It supports multiple interface extraction algorithms, several geometric fitting strategies, explicit or automatic wall detection, and both per-frame and pooled analyses through a fully modular analysis pipeline.

Supported trajectory formats include **LAMMPS**, **XYZ**, and **ASE**.

<p align="center"> <img src="wetting_angle_kit/docs/images/pipeline_wak_3component.png" width="900"> </p>


## How the methods work

### Interface extraction: turning atoms into a surface

In an MD simulation, the liquid-vapor interface isn't a sharp line — density drops smoothly over about 1 Å. Interface extraction recovers a clean set of interface points from that noisy atom cloud.

It's built from two independent choices: a `SpaceSampling` (*where* to evaluate density) and a `DensityEstimator` (*how* to evaluate it). An `InterfaceExtractor` combines one of each, and any combination is valid.

- **`SpaceSampling.rays(...)`** sends a fan of rays out from the droplet's center. The interface point along each ray is the half-density point of a 1D tanh fit. The fan can be azimuthal slices in the `(x, z)` plane, or a Fibonacci sphere of directions for a whole-shape fit.
- **`SpaceSampling.grid(...)`** builds a fixed grid in space, then traces the half-density contour using marching squares (slicing mode) or marching cubes (whole mode). In slicing mode it iterates per slice, so each slice gets its own `(s, z)` contour, surfacing per-slice asymmetry. This works well when a single frame doesn't have many atoms.
- **`DensityEstimator.gaussian(density_sigma=…)`** is a smooth 3D Gaussian KDE, with no per-cell noise.
- **`DensityEstimator.binning(bin_width=…)`** is a cheap 3D histogram. `bin_width` is only required for ray sampling, where it sets the kernel size.
### Surface fitting: choosing a shape to fit

- **Slicing fit** fits a circle independently in each slice's `(x, z)` plane, then averages the per-slice angles. Good for droplets that aren't perfectly spherical — the spread across slices gives you a natural `±σ`.
- **Whole fit** fits one sphere (spherical droplets) or cylinder (cylindrical droplets) to the full 3D interface. Uses the Taubin method, with optional bootstrap resampling for an uncertainty estimate.
- **Coupled fit** solves everything at once: a 7-parameter (2D) or 9-parameter (3D) tanh density model fits the interface, the wall plane, and the cap geometry together in a single nonlinear least-squares fit. Density per cell comes from a `DensityEstimator` — histogram (`binning()`, default) or Gaussian KDE (`gaussian(density_sigma=…)`, smoother, no Poisson noise). Most statistically efficient when you pool many frames into one fit.
### Wall detection: locating the wall plane

Contact angle is measured where the cap meets the wall, so the wall plane needs to be found:

- **`min_plus_offset`** — derive the wall from the interface itself (lowest point + offset). Works when interface points actually reach the wall.
- **`from_atoms`** — read real wall-atom positions from the trajectory, placing the wall at the mean of the top atomic layer. Most physically accurate when the simulation includes substrate atoms.
- **`explicit`** — you supply the wall's z-position directly, if it's already known from the simulation setup.
### Frame batching: one angle per frame, or per batch

`TemporalAggregator` groups trajectory frames before fitting. `batch_size=1` gives one angle per frame (an angle-vs-time curve). `batch_size=N` pools N frames per fit (less noise, less time resolution). `batch_size=-1` pools every frame into one fit.

## Two entry points

1. **`TrajectoryAnalyzer`** — combines the four pieces above (`InterfaceExtractor` × `SurfaceFitter` × `WallDetector` × `TemporalAggregator`). Use this for per-frame time resolution, or to mix and match approaches.
2. **`CoupledFit2DAnalyzer` / `CoupledFit3DAnalyzer`** — the joint-fit shortcut. One angle per pooled batch from the tanh density model. The density estimator is still pluggable (`binning()` or `gaussian(...)`). Best with many frames and no need for per-frame resolution.
Full docs, with worked examples and tutorials, are [here](https://matgenix.github.io/wetting-angle-kit).

## Installation

### Prerequisites

Before installing wetting-angle-kit, ensure you have **Python 3.10 or higher**
installed on your system.

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

OVITO must be installed first using pip:

```sh
pip install  ovito==3.11.3
```

## Quick Start


```python
from wetting_angle_kit.analysis import (
    CoupledFit2DAnalyzer,
    DensityEstimator,
    InterfaceExtractor,
    SpaceSampling,
    SurfaceFitter,
    TrajectoryAnalyzer,
    WallDetector,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.parsers import XYZParser, XYZWaterFinder

trajectory_file = "trajectory.xyz"

# Identify water oxygen atoms by neighbour count.
finder = XYZWaterFinder(trajectory_file)
oxygen_ids = finder.get_water_oxygen_indices(frame_index=0)

parser = XYZParser(trajectory_file)

# --- Composable pipeline (per-frame slicing-fit angles) ---
slicing = TrajectoryAnalyzer(
    parser=parser,
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
    interface_extractor=InterfaceExtractor(
        sampling=SpaceSampling.rays(
            delta_azimuthal=5.0,  # 5° between slicing planes
            delta_polar=8.0,
        ),
        density=DensityEstimator.gaussian(),
    ),
    surface_fitter=SurfaceFitter.slicing(surface_filter_offset=2.0),
    wall_detector=WallDetector.min_plus_offset(offset=0.0),
    temporal_aggregator=TemporalAggregator(batch_size=1),  # one angle per frame
)
results = slicing.analyze(range(0, 24))
print(results.mean_angle, results.std_angle)

# --- Joint coupled-fit (one robust angle over a pooled batch) ---
coupled_fit = CoupledFit2DAnalyzer(
    parser=parser,
    atom_indices=oxygen_ids,
    droplet_geometry="spherical",
    binning_params={
        "xi_0": 0.0, "xi_f": 70.0, "bin_width_x": 2.0,
        "zi_0": 0.0, "zi_f": 70.0, "bin_width_z": 2.0,
    },
    # Default: histogram density. Swap in `DensityEstimator.gaussian(
    # density_sigma=2.5)` for a smooth Gaussian-KDE density field
    # useful on per-frame batches or sparse systems.
    density_estimator=DensityEstimator.binning(),
)
results_coupled_fit = coupled_fit.analyze(range(0, 200))
print(results_coupled_fit.mean_angle, results_coupled_fit.std_angle)
```
