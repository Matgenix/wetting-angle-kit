"""Edge-case tests for the trajectory analyzer pipeline.

Phase 11 migration: the legacy ``SlicingFrameFitter``-internal tests
(``find_intersection``, ``calculate_y_axis_list``, etc.) no longer have
direct analogues in the new architecture (the per-slice helpers live
inside ``_SlicingFitter`` / ``_RaysGaussianExtractor`` and are exercised
by the Phase 2–4 tests). The only edge case worth keeping in this file
is the parser-extension validation, which the new ``TrajectoryAnalyzer``
inherits from the shared ``_BatchedTrajectoryAnalyzer`` base.
"""

import numpy as np
import pytest

from wetting_angle_kit.analysis import (
    InterfaceExtractor,
    SurfaceFitter,
    TrajectoryAnalyzer,
)


def test_unsupported_extension_raises_at_construction(tmp_path) -> None:
    """Unknown trajectory extension must fail fast at construction.

    The shared ``_BatchedTrajectoryAnalyzer`` calls
    ``detect_parser_type(parser.filepath)`` in ``__init__`` for the same
    reason the legacy code did: the actual parser is rebuilt inside
    worker processes, where a parser-type error would be silently
    swallowed.
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
            interface_extractor=InterfaceExtractor.rays_gaussian(
                delta_azimuthal=20.0, delta_polar=8.0
            ),
            surface_fitter=SurfaceFitter.slicing(),
        )
