"""Public exports for binning contact angle method."""

from wetting_angle_kit.analysis.binning.angle_fitting import (
    ContactAngleBinning,
)
from wetting_angle_kit.analysis.binning.surface_definition import (
    HyperbolicTangentModel,
)

__all__ = ["ContactAngleBinning", "HyperbolicTangentModel"]
