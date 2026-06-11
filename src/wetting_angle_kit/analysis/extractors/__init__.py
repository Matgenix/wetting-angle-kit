"""Interface extractors: build the interface point set from raw atoms.

An :class:`InterfaceExtractor` converts pooled liquid-atom coordinates
into one of two output shapes, determined by the :class:`SurfaceFitter`
the analyzer is paired with:

- ``surface_kind="slicing"`` → a list of per-slice ``(M_i, 2)`` arrays
  in the slice ``(x, z)`` plane;
- ``surface_kind="whole"`` → a single ``(N, 3)`` shell array in the
  internal ``(x, y, z)`` frame.

Extractors are constructed through classmethod factories on the base
class — each factory configures one sampling + density-kernel
combination::

    InterfaceExtractor.rays_gaussian(...)   # ray fan + Gaussian KDE + tanh
    InterfaceExtractor.rays_binning(...)    # ray fan + histogram bins + tanh
    InterfaceExtractor.grid_gaussian(...)   # 2D KDE map + isocontour (slicing only)
    InterfaceExtractor.grid_binning(...)    # 2D histogram map + isocontour
                                            # (slicing only)

The pairing between the chosen extractor and the analyzer's
:class:`SurfaceFitter` is validated at :class:`TrajectoryAnalyzer`
construction via :meth:`InterfaceExtractor.validate_compatibility`.
"""

from wetting_angle_kit.analysis.extractors.base import InterfaceExtractor

__all__ = ["InterfaceExtractor"]
