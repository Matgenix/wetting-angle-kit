"""Public exports for the sliced contact angle method."""

from wetting_angle_kit.contact_angle_method.sliced_method.angle_fitting_sliced import (
    ContactAngleSliced,
)
from wetting_angle_kit.contact_angle_method.sliced_method.multi_processing import (
    ContactAngleSlicedParallel,
)
from wetting_angle_kit.contact_angle_method.sliced_method.surface_defined import (
    SurfaceDefinition,
)

__all__ = [
    "ContactAngleSliced",
    "ContactAngleSlicedParallel",
    "SurfaceDefinition",
]
