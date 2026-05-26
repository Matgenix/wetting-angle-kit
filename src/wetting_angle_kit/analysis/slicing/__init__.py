"""Public exports for the slicing contact angle method."""

from wetting_angle_kit.analysis.slicing.angle_fitting import (
    ContactAngleSlicing,
)
from wetting_angle_kit.analysis.slicing.parallel import (
    ContactAngleSlicingParallel,
)
from wetting_angle_kit.analysis.slicing.surface_definition import (
    SurfaceDefinition,
)

__all__ = [
    "ContactAngleSlicing",
    "ContactAngleSlicingParallel",
    "SurfaceDefinition",
]
