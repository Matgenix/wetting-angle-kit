"""Edge-case tests for the trajectory-analyzer pipeline.

The per-slice geometry helpers are exercised by the main fitter and
extractor test suites; what's kept here is the parser-extension
validation that :class:`TrajectoryAnalyzer` inherits from the
shared :class:`_BatchedTrajectoryAnalyzer` base.
"""

import numpy as np
import pytest

from wetting_angle_kit.analysis import (
    DensityEstimator,
    InterfaceExtractor,
    SpaceSampling,
    SurfaceFitter,
    TrajectoryAnalyzer,
)


def test_unsupported_extension_raises_at_construction(tmp_path) -> None:
    """Unknown trajectory extension must fail fast at construction.

    The shared ``_BatchedTrajectoryAnalyzer`` calls
    ``detect_parser_type(parser.filepath)`` in ``__init__`` because the
    actual parser is rebuilt inside worker processes, where a
    parser-type error would otherwise be silently swallowed.
    """
    fake = tmp_path / "trajectory.bogus"
    fake.write_text("not a real trajectory\n")

    class _FakeParser:
        filepath = str(fake)

    with pytest.raises(ValueError, match="Unsupported trajectory file format"):
        TrajectoryAnalyzer(
            parser=_FakeParser(),
            atom_indices=np.array([]),
            droplet_geometry="spherical",
            interface_extractor=InterfaceExtractor(
                sampling=SpaceSampling.rays(delta_azimuthal=20.0, delta_polar=8.0),
                density=DensityEstimator.gaussian(),
            ),
            surface_fitter=SurfaceFitter.slicing(),
        )
