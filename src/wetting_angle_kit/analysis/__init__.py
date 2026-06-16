"""Contact-angle analysis orchestrators and per-method engines.

Public API summary
------------------

Top-level analyzers (call ``analyze()`` to run a study)::

    TrajectoryAnalyzer         # decomposed pipeline: extractor → wall → fitter
    CoupledFit2DAnalyzer   # coupled fit on a 2D density grid
    CoupledFit3DAnalyzer   # coupled fit on a 3D density grid (spherical only)

Strategy components (compose into :class:`TrajectoryAnalyzer`)::

    DropletGeometry      # spherical / cylinder_x / cylinder_y
    TemporalAggregator   # per-frame / pooled batches
    InterfaceExtractor   # composes (sampling, density)
    SpaceSampling        # rays / grid
    DensityEstimator     # gaussian / binning
    SurfaceFitter        # slicing / whole
    WallDetector         # min_plus_offset / explicit / from_atoms

The coupled-fit analyzers also accept a :class:`DensityEstimator`
directly (they have their own grid / projection logic).

Results dataclasses returned by ``analyze()``::

    TrajectoryResults, BatchResult, SlicingBatchResult, WholeBatchResult
    CoupledFit2DResults, CoupledFit2DBatchResult
    CoupledFit3DResults, CoupledFit3DBatchResult
"""

from wetting_angle_kit.analysis._base import BaseTrajectoryAnalyzer

# Top-level analyzers.
from wetting_angle_kit.analysis.coupled_fit import DensityEstimator
from wetting_angle_kit.analysis.coupled_fit.analyzer_2d import (
    CoupledFit2DAnalyzer,
)
from wetting_angle_kit.analysis.coupled_fit.analyzer_3d import (
    CoupledFit3DAnalyzer,
)
from wetting_angle_kit.analysis.fitters import (
    FitOutput,
    SlicingFitOutput,
    SurfaceFitter,
    WholeFitOutput,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry

# Strategy components.
from wetting_angle_kit.analysis.interface import (
    InterfaceExtractor,
    SpaceSampling,
)

# Results dataclasses.
from wetting_angle_kit.analysis.results import (
    BatchResult,
    CoupledFit2DBatchResult,
    CoupledFit2DResults,
    CoupledFit3DBatchResult,
    CoupledFit3DResults,
    SlicingBatchResult,
    TrajectoryResults,
    WholeBatchResult,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.analysis.trajectory import TrajectoryAnalyzer
from wetting_angle_kit.analysis.wall import WallDetector

__all__ = [
    # Top-level analyzers.
    "BaseTrajectoryAnalyzer",
    "TrajectoryAnalyzer",
    "CoupledFit2DAnalyzer",
    "CoupledFit3DAnalyzer",
    # Strategy components.
    "DensityEstimator",
    "DropletGeometry",
    "TemporalAggregator",
    "InterfaceExtractor",
    "SpaceSampling",
    "SurfaceFitter",
    "FitOutput",
    "SlicingFitOutput",
    "WholeFitOutput",
    "WallDetector",
    # Results.
    "BatchResult",
    "SlicingBatchResult",
    "WholeBatchResult",
    "TrajectoryResults",
    "CoupledFit2DBatchResult",
    "CoupledFit2DResults",
    "CoupledFit3DBatchResult",
    "CoupledFit3DResults",
]
