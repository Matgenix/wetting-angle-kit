from typing import Any

from wetting_angle_kit.contact_angle_method.contact_angle_analyzer import (
    BaseContactAngleAnalyzer,
    BinningContactAngleAnalyzer,
    SlicedContactAngleAnalyzer,
)


def contact_angle_analyzer(
    method: str, parser: Any, output_dir: str, **kwargs: Any
) -> BaseContactAngleAnalyzer:
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
