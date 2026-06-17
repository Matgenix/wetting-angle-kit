# wetting-angle-kit

[![tests](https://img.shields.io/github/actions/workflow/status/Matgenix/wetting-angle-kit/testing.yml?branch=main&label=tests)](https://github.com/Matgenix/wetting-angle-kit/actions/workflows/testing.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/Matgenix/wetting-angle-kit/deploy-docs.yml?branch=main&label=docs)](https://github.com/Matgenix/wetting-angle-kit/actions/workflows/deploy-docs.yml)
[![code coverage](https://codecov.io/gh/Matgenix/wetting-angle-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/Matgenix/wetting-angle-kit)
[![pypi version](https://img.shields.io/pypi/v/wetting-angle-kit?color=blue)](https://pypi.org/project/wetting-angle-kit/)
[![Python versions](https://img.shields.io/pypi/pyversions/wetting-angle-kit)](https://pypi.org/project/wetting-angle-kit/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-matgenix.github.io-blue)](https://matgenix.github.io/wetting-angle-kit)

wetting-angle-kit parses MD trajectories (LAMMPS dump, XYZ, ASE) and computes the contact angle of a droplet sitting on a planar wall. The package follows the same conceptual recipe every method uses — extract the liquid-vapor interface from atom positions, decide where the wall plane sits, fit a geometric shape, read off the angle from the shape/wall intersection — but exposes each step as a swappable component so users can match the method to their system.

## How the methods are built

### Interface extraction: how do we turn atoms into a surface?

The liquid-vapor interface isn't a sharp surface in an MD simulation — the density drops smoothly over ~1 Å. Two extraction strategies recover a clean set of interface points from the noisy atom cloud:

The package exposes two orthogonal strategy axes for interface extraction. A `SpaceSampling` decides *where* density is evaluated; a `DensityEstimator` decides *how*. An `InterfaceExtractor` composes one of each.

- **Sampling: `SpaceSampling.rays(...)`** emits a fan of rays from the droplet centre of mass; the interface along each ray is the half-density point of a 1D tanh fit. The fan layout is azimuthal slices in the `(x, z)` plane (for a per-slice fit) or a Fibonacci sphere of directions (for a whole-shape fit).
- **Sampling: `SpaceSampling.grid(...)`** builds a fixed-cell grid in space; the interface is the iso-density contour at the half-bulk level, traced via marching squares (slicing mode) or marching cubes (whole mode). In slicing mode the grid iterates per slice (per azimuthal angle for spherical droplets, per axial step for cylinder droplets), so the slicing fit sees one `(s, z)` contour per slice and can expose per-slice asymmetry. Closer to the "average over many frames" intuition than ray fans; works well when atom statistics are limited per frame.
- **Density: `DensityEstimator.gaussian(density_sigma=…)`** is a 3D Gaussian KDE (smooth, no per-cell Poisson noise).
- **Density: `DensityEstimator.binning(bin_width=…)`** is a 3D top-hat histogram (cheap; bin_width required only for the rays sampling, where it sets the pointwise kernel size).

Any sampling × any density is a valid extractor.

### Surface fitting: what geometric shape do we fit to those points?

- **Slicing fit** — independently fits an algebraic circle in each slice's `(x, z)` plane, then averages the per-slice contact angles. Good when the droplet might be slightly non-spherical: the per-slice scatter naturally reports a `±σ` band.
- **Whole fit** — fits a single sphere (spherical droplet) or cylinder (cylindrical droplet) to the entire 3D interface shell. Uses the algebraic Taubin method, plus optional bootstrap resampling to put an uncertainty on the recovered angle.
- **Coupled fit** (joint approach) — a 7-parameter (2D) or 9-parameter (3D) hyperbolic-tangent density model that solves "where is the interface", "where is the wall plane", and "what's the cap geometry" in one nonlinear least-squares fit on a density field. The per-cell density is computed by a pluggable `DensityEstimator` strategy: a top-hat histogram (`DensityEstimator.binning()`, default) or a 3D Gaussian KDE evaluated at the cell centres (`DensityEstimator.gaussian(density_sigma=…)`); the KDE variant trades a small constant cost for a smooth, Poisson-noise-free density. Statistically efficient when you pool many frames per batch.

### Wall detection: where is the wall plane?

The contact angle is measured at the cap–wall intersection, so the wall plane has to be located explicitly:

- `min_plus_offset`: derive the wall from the interface itself (lowest interface point + offset). Works for slicing geometries and full-sphere ray fans, where the interface points reach the wall.
- `from_atoms`: read the actual wall atom positions from the trajectory and place the wall at the mean of the top atomic layer. Most physically faithful when the simulation explicitly contains substrate atoms.
- `explicit`: caller supplies the wall z directly — useful when the wall position is known a priori from the simulation setup.

### Frame batching: per-frame angle or pooled batch?

The `TemporalAggregator` groups trajectory frames into batches before fitting. `batch_size=1` runs the full pipeline once per frame (giving you an angle vs time curve); `batch_size=N` pools `N` frames together and fits one angle per pool (more atoms per fit → less noise, less time resolution); `batch_size=-1` pools everything into a single batch.

## Two top-level entry points

1. **`TrajectoryAnalyzer`** — composes the four strategies above (`InterfaceExtractor` × `SurfaceFitter` × `WallDetector` × `TemporalAggregator`). Use it when you want per-frame time resolution or when you want to mix-and-match approaches (e.g. ray-fan extractor + whole-fit + explicit wall + 5-frame batches).
2. **`CoupledFit2DAnalyzer` / `CoupledFit3DAnalyzer`** — the joint-fit alternative. One robust angle per pooled batch via the hyperbolic-tangent density model. The per-cell density estimator is pluggable (`DensityEstimator.binning()` or `DensityEstimator.gaussian(...)`). Best when you have many frames and don't need per-frame time resolution.

The documentation is available [here](https://matgenix.github.io/wetting-angle-kit), with worked examples and tutorials.

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
results = slicing.analyze(range(0, 50))
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
    # density_sigma=2.5)` for a smooth Gaussian-KDE density field —
    # useful on per-frame batches or sparse systems.
    density_estimator=DensityEstimator.binning(),
)
results_coupled_fit = coupled_fit.analyze(range(0, 200))
print(results_coupled_fit.mean_angle, results_coupled_fit.std_angle)
```
