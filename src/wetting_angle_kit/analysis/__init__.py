"""Contact-angle analysis orchestrators and per-method engines.

Public API summary
------------------

Top-level analyzers (call ``analyze()`` to run a study)::

    TrajectoryAnalyzer         # decomposed pipeline: extractor → wall → fitter
    CoupledBinning2DAnalyzer   # joint fit on a 2D density grid
    CoupledBinning3DAnalyzer   # joint fit on a 3D density grid (spherical only)

Strategy components (compose into :class:`TrajectoryAnalyzer`)::

    DropletGeometry      # spherical / cylinder_x / cylinder_y
    TemporalAggregator   # per-frame / pooled batches
    InterfaceExtractor   # rays / grid × gaussian / binning
    SurfaceFitter        # slicing / whole
    WallDetector         # min_plus_offset / explicit / from_atoms

Results dataclasses returned by ``analyze()``::

    TrajectoryResults, BatchResult, SlicingBatchResult, WholeBatchResult
    CoupledBinning2DResults, CoupledBinning2DBatchResult
    CoupledBinning3DResults, CoupledBinning3DBatchResult
"""

from wetting_angle_kit.analysis.analyzer import BaseTrajectoryAnalyzer

# Top-level analyzers.
from wetting_angle_kit.analysis.coupled_binning_2d import CoupledBinning2DAnalyzer
from wetting_angle_kit.analysis.coupled_binning_3d import CoupledBinning3DAnalyzer

# Strategy components.
from wetting_angle_kit.analysis.extractors import InterfaceExtractor
from wetting_angle_kit.analysis.fitters import (
    FitOutput,
    SlicingFitOutput,
    SurfaceFitter,
    WholeFitOutput,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry

# Results dataclasses.
from wetting_angle_kit.analysis.results import (
    BatchResult,
    CoupledBinning2DBatchResult,
    CoupledBinning2DResults,
    CoupledBinning3DBatchResult,
    CoupledBinning3DResults,
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
    "CoupledBinning2DAnalyzer",
    "CoupledBinning3DAnalyzer",
    # Strategy components.
    "DropletGeometry",
    "TemporalAggregator",
    "InterfaceExtractor",
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
    "CoupledBinning2DBatchResult",
    "CoupledBinning2DResults",
    "CoupledBinning3DBatchResult",
    "CoupledBinning3DResults",
]
