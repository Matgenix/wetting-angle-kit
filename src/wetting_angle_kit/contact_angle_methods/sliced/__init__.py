"""Public exports for the sliced contact angle method."""

from wetting_angle_kit.contact_angle_methods.sliced.angle_fitting import (
    ContactAngleSliced,
)
from wetting_angle_kit.contact_angle_methods.sliced.parallel import (
    ContactAngleSlicedParallel,
)
from wetting_angle_kit.contact_angle_methods.sliced.surface_definition import (
    SurfaceDefinition,
)

__all__ = [
    "ContactAngleSliced",
    "ContactAngleSlicedParallel",
    "SurfaceDefinition",
]
