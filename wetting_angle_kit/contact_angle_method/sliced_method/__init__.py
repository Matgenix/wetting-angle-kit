"""Public exports for sliced contact angle method."""

from .angle_fitting_sliced import (
    ContactAngleSliced as _ContactAngleSliced,
)
from .multi_processing import (
    ContactAngleSlicedParallel as _ContactAngleSlicedParallel,
)
from .surface_defined import SurfaceDefinition as _SurfaceDefinition

__all__ = [
    "ContactAngleSliced",
    "ContactAngleSliced",
    "ContactAngleSlicedParallel",
    "SurfaceDefinition",
]

ContactAngleSliced = _ContactAngleSliced
ContactAngleSliced = _ContactAngleSliced
ContactAngleSlicedParallel = _ContactAngleSlicedParallel
SurfaceDefinition = _SurfaceDefinition
