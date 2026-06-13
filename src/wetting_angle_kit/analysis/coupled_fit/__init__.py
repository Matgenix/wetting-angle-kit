"""Coupled-fit joint contact-angle analyzers.

Two top-level analyzers that solve interface extraction, wall
detection, and surface fit jointly via a hyperbolic-tangent density
model:

- :class:`CoupledFit2DAnalyzer` — seven-parameter fit on a 2D
  ``(xi, zi)`` density grid (radial symmetry assumption).
- :class:`CoupledFit3DAnalyzer` — nine-parameter fit on a full 3D
  ``(xi, yi, zi)`` density grid (no symmetry assumption; spherical
  droplets only — cylinder droplets are rejected at construction).

Both analyzers accept a :class:`DensityEstimator` strategy that
controls how the per-cell density is computed from pooled atom
positions: :meth:`DensityEstimator.binning` (top-hat histogram, the
default) or :meth:`DensityEstimator.gaussian` (3D Gaussian KDE on
the cell centres). Switching to the Gaussian variant trades a small
constant cost per fit for a smooth density field with no per-cell
Poisson noise.

Use these analyzers when you have many frames per batch and want a
single robust estimate; use :class:`TrajectoryAnalyzer` with
separable strategies for per-frame time resolution.
"""

from wetting_angle_kit.analysis.coupled_fit.analyzer_2d import (
    CoupledFit2DAnalyzer,
)
from wetting_angle_kit.analysis.coupled_fit.analyzer_3d import (
    CoupledFit3DAnalyzer,
)
from wetting_angle_kit.analysis.density_estimator import (
    DensityEstimator,
)

__all__ = [
    "CoupledFit2DAnalyzer",
    "CoupledFit3DAnalyzer",
    "DensityEstimator",
]
