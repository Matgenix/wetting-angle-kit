"""Contact-angle analysis orchestrators and per-method engines."""

from typing import Any

from wetting_angle_kit.analysis.analyzer import (
    BaseContactAngleAnalyzer,
    BinningContactAngleAnalyzer,
    SlicingContactAngleAnalyzer,
)
from wetting_angle_kit.analysis.binning.angle_fitting import (
    ContactAngleBinning,
)
from wetting_angle_kit.analysis.slicing.angle_fitting import (
    ContactAngleSlicing,
)
from wetting_angle_kit.analysis.slicing.parallel import (
    ContactAngleSlicingParallel,
)


def contact_angle_analyzer(
    method: str, parser: Any, output_dir: str, **kwargs: Any
) -> BaseContactAngleAnalyzer:
    """Return an analyzer instance for the requested contact-angle method.

    Parameters
    ----------
    method : str
        Analysis method; one of ``"slicing"`` or ``"binning"``.
    parser : BaseParser
        Trajectory parser instance.
    output_dir : str
        Directory for output files.
    **kwargs
        Forwarded to the selected analyzer constructor.

    Returns
    -------
    BaseContactAngleAnalyzer
        Configured analyzer ready to call ``analyze()``.
    """
    if method == "slicing":
        return SlicingContactAngleAnalyzer(
            parser=parser, output_dir=output_dir, **kwargs
        )
    elif method == "binning":
        return BinningContactAngleAnalyzer(
            parser=parser, output_dir=output_dir, **kwargs
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'slicing' or 'binning'.")


__all__ = [
    "BaseContactAngleAnalyzer",
    "SlicingContactAngleAnalyzer",
    "BinningContactAngleAnalyzer",
    "contact_angle_analyzer",
    "ContactAngleBinning",
    "ContactAngleSlicing",
    "ContactAngleSlicingParallel",
]
