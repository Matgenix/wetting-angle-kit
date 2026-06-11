"""Coupled-binning joint contact-angle analyzers.

Two top-level analyzers that solve interface extraction, wall
detection, and surface fit jointly via a hyperbolic-tangent density
model:

- :class:`CoupledBinning2DAnalyzer` — seven-parameter fit on a 2D
  ``(xi, zi)`` density grid (radial symmetry assumption).
- :class:`CoupledBinning3DAnalyzer` — nine-parameter fit on a full 3D
  ``(xi, yi, zi)`` density grid (no symmetry assumption; spherical
  droplets only — cylinder droplets are rejected at construction).

Use these when you have many frames per batch and want a single robust
estimate; use :class:`TrajectoryAnalyzer` with separable strategies
for per-frame time resolution.
"""

from wetting_angle_kit.analysis.coupled_binning.analyzer_2d import (
    CoupledBinning2DAnalyzer,
)
from wetting_angle_kit.analysis.coupled_binning.analyzer_3d import (
    CoupledBinning3DAnalyzer,
)

__all__ = [
    "CoupledBinning2DAnalyzer",
    "CoupledBinning3DAnalyzer",
]
