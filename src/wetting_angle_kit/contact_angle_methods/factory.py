from typing import Any

from wetting_angle_kit.contact_angle_methods.analyzer import (
    BaseContactAngleAnalyzer,
    BinningContactAngleAnalyzer,
    SlicedContactAngleAnalyzer,
)


def contact_angle_analyzer(
    method: str, parser: Any, output_dir: str, **kwargs: Any
) -> BaseContactAngleAnalyzer:
    """Return an analyzer instance for the requested contact-angle method.

    Parameters
    ----------
    method : str
        Analysis method; one of ``"sliced"`` or ``"binning"``.
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
    if method == "sliced":
        return SlicedContactAngleAnalyzer(
            parser=parser, output_dir=output_dir, **kwargs
        )
    elif method == "binning":
        return BinningContactAngleAnalyzer(
            parser=parser, output_dir=output_dir, **kwargs
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'sliced' or 'binning'.")
