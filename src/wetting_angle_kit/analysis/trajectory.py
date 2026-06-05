"""The decomposed :class:`TrajectoryAnalyzer`.

:class:`TrajectoryAnalyzer` ties together the five strategy components
that define a contact-angle analysis pipeline:

- :class:`DropletGeometry` — droplet symmetry / internal axis layout
- :class:`TemporalAggregator` — per-frame vs pooled-batch scheduling
- :class:`InterfaceExtractor` — atom → interface points
- :class:`SurfaceFitter` — interface points → contact angle
- :class:`WallDetector` — wall plane location

The class extends the shared :class:`_BatchedTrajectoryAnalyzer`
worker-pool scaffolding by implementing the four extension points
documented there. The per-batch wiring lives in
:meth:`_process_batch_worker`; algorithm bodies (extraction, fitting,
wall detection beyond :meth:`WallDetector.min_plus_offset`) remain
``NotImplementedError`` until porting begins.

The joint-fit analyzers (:class:`CoupledBinning2DAnalyzer`,
:class:`CoupledBinning3DAnalyzer`) live in their own modules and
share only the worker-pool scaffolding, not this strategy pipeline.
"""

import logging
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis.base import (
    _BatchedTrajectoryAnalyzer,
    build_parser,
    gather_batch_coords,
    gather_wall_coords,
)
from wetting_angle_kit.analysis.extractors import InterfaceExtractor
from wetting_angle_kit.analysis.fitters import SurfaceFitter
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import BatchResult, TrajectoryResults
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.analysis.wall import WallContext, WallDetector

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer(_BatchedTrajectoryAnalyzer):
    """Decomposed contact-angle analyzer: extractor → wall → fitter.

    Parameters
    ----------
    parser : BaseParser
        Trajectory parser instance. Only ``parser.filepath`` and
        ``parser.frame_count()`` are read in the parent process; each
        worker rebuilds its own parser from ``filepath``.
    atom_indices : ndarray, optional
        Indices of the liquid atoms; passed through to the parser's
        per-frame ``parse`` call. Empty by default.
    droplet_geometry : DropletGeometry or str, default ``"spherical"``
        Either a :class:`DropletGeometry` instance or the bare name
        string. Drives the internal axis convention and the per-slice
        layout used by the extractor.
    interface_extractor : InterfaceExtractor
        Built via one of :meth:`InterfaceExtractor.rays_gaussian` /
        :meth:`InterfaceExtractor.rays_binning` /
        :meth:`InterfaceExtractor.grid_gaussian` /
        :meth:`InterfaceExtractor.grid_binning`.
    surface_fitter : SurfaceFitter
        Built via :meth:`SurfaceFitter.slicing` or
        :meth:`SurfaceFitter.whole`. Its :attr:`kind` must match the
        extractor's natural output, which is enforced via
        :meth:`InterfaceExtractor.validate_compatibility` at
        construction.
    wall_detector : WallDetector, optional
        Built via :meth:`WallDetector.min_plus_offset` /
        :meth:`WallDetector.explicit` /
        :meth:`WallDetector.from_atoms`. Defaults to
        ``WallDetector.min_plus_offset(offset=2.0)``.
    temporal_aggregator : TemporalAggregator, optional
        Defaults to per-frame analysis (``batch_size=1``).
    precentered : bool, default ``False``
        Skip per-frame circular-mean PBC recentering. Setting this on
        a trajectory that does NOT satisfy the precondition will
        produce wrong results.
    wall_atom_indices : ndarray, optional
        Required when ``wall_detector`` is a
        :meth:`WallDetector.from_atoms` instance. The analyzer gathers
        and pools these coordinates per batch and supplies them to the
        detector via :attr:`WallContext.wall_coords`.
    """

    #: Per-process worker state — shadowed from the parent so this
    #: subclass writes to its own slot and never collides with the
    #: coupled-fit analyzers in the same process.
    _WORKER_STATE: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        parser: Any,
        atom_indices: np.ndarray | None = None,
        droplet_geometry: DropletGeometry | str = "spherical",
        *,
        interface_extractor: InterfaceExtractor,
        surface_fitter: SurfaceFitter,
        wall_detector: WallDetector | None = None,
        temporal_aggregator: TemporalAggregator | None = None,
        precentered: bool = False,
        wall_atom_indices: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            parser=parser,
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            temporal_aggregator=temporal_aggregator,
            precentered=precentered,
            wall_atom_indices=wall_atom_indices,
        )
        self.interface_extractor = interface_extractor
        self.surface_fitter = surface_fitter
        self.wall_detector = wall_detector or WallDetector.min_plus_offset(offset=2.0)
        # Fail fast on incompatible component combinations. The
        # extractor validates against (surface_kind, droplet_geometry);
        # the fitter validates against (droplet_geometry).
        self.interface_extractor.validate_compatibility(
            surface_kind=self.surface_fitter.kind,
            droplet_geometry=self.droplet_geometry,
        )
        self.surface_fitter.validate_compatibility(self.droplet_geometry)

    # ------------------------------------------------------------------
    # _BatchedTrajectoryAnalyzer extension points.
    # ------------------------------------------------------------------

    def _tqdm_desc(self) -> str:
        return (
            f"TrajectoryAnalyzer ({self.surface_fitter.kind} / "
            f"{self.interface_extractor.sampling})"
        )

    def _init_args(self) -> tuple:
        return (
            self.parser.filepath,
            self.atom_indices,
            self.wall_atom_indices,
            self.droplet_geometry,
            self.interface_extractor,
            self.surface_fitter,
            self.wall_detector,
            self.precentered,
        )

    @staticmethod
    def _init_worker(
        filename: str,
        atom_indices: np.ndarray,
        wall_atom_indices: np.ndarray | None,
        droplet_geometry: DropletGeometry,
        interface_extractor: InterfaceExtractor,
        surface_fitter: SurfaceFitter,
        wall_detector: WallDetector,
        precentered: bool,
    ) -> None:
        cls = TrajectoryAnalyzer
        cls._WORKER_STATE.clear()
        cls._WORKER_STATE.update(
            parser=build_parser(filename),
            atom_indices=atom_indices,
            wall_atom_indices=wall_atom_indices,
            droplet_geometry=droplet_geometry,
            interface_extractor=interface_extractor,
            surface_fitter=surface_fitter,
            wall_detector=wall_detector,
            precentered=precentered,
        )

    @staticmethod
    def _process_batch_worker(frame_indices: list[int]) -> BatchResult | None:
        state = TrajectoryAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        wall_atom_indices: np.ndarray | None = state["wall_atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        extractor: InterfaceExtractor = state["interface_extractor"]
        fitter: SurfaceFitter = state["surface_fitter"]
        detector: WallDetector = state["wall_detector"]
        precentered: bool = state["precentered"]
        try:
            coords, center, max_dist = gather_batch_coords(
                parser=parser,
                frame_indices=frame_indices,
                atom_indices=atom_indices,
                droplet_geometry=droplet_geometry,
                precentered=precentered,
            )
            wall_coords = (
                gather_wall_coords(
                    parser=parser,
                    frame_indices=frame_indices,
                    wall_atom_indices=wall_atom_indices,
                    droplet_geometry=droplet_geometry,
                )
                if wall_atom_indices is not None
                else None
            )
            interface_data = extractor.extract(
                liquid_coordinates=coords,
                center_geom=center,
                droplet_geometry=droplet_geometry,
                max_dist=max_dist,
                surface_kind=fitter.kind,
            )
            z_wall = detector.detect(
                WallContext(
                    interface_data=interface_data,
                    wall_coords=wall_coords,
                )
            )
            fit_output = fitter.fit(
                interface_data=interface_data,
                z_wall=z_wall,
                droplet_geometry=droplet_geometry,
            )
            return fit_output.to_batch_result(list(frame_indices))
        except Exception as e:
            logger.error(f"Error processing batch {frame_indices}: {e}", exc_info=True)
            return None

    def _build_results(self, batches: list[BatchResult]) -> TrajectoryResults:
        return TrajectoryResults(
            batches=batches,
            method_metadata={
                "kind": self.surface_fitter.kind,
                "sampling": self.interface_extractor.sampling,
                "droplet_geometry": self.droplet_geometry.name,
                "batch_size": self.temporal_aggregator.batch_size,
            },
        )
