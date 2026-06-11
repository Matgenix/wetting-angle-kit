"""Surface fitters: derive a contact angle from interface points + wall.

A :class:`SurfaceFitter` consumes an interface point set produced by an
:class:`InterfaceExtractor` plus a wall z-coordinate produced by a
:class:`WallDetector`, and returns one :class:`BatchResult` per call
holding the contact angle and fit diagnostics.

Two fitter kinds are supported:

- ``slicing``: one algebraic-circle fit per slice in the slice's
  ``(x, z)`` plane, then a mean across slices.
- ``whole``: one algebraic-sphere fit (spherical droplet) or
  algebraic-cylinder fit (cylindrical droplet) to the 3D interface
  shell.

Users construct fitters through classmethod factories on the base
class::

    SurfaceFitter.slicing()
    SurfaceFitter.whole(bootstrap_samples=0)
"""

from wetting_angle_kit.analysis.fitters.base import (
    FitOutput,
    SlicingFitOutput,
    SurfaceFitter,
    WholeFitOutput,
)

__all__ = [
    "SurfaceFitter",
    "FitOutput",
    "SlicingFitOutput",
    "WholeFitOutput",
]
