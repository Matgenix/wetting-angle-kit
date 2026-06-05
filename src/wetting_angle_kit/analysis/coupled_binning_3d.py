"""Coupled 3D-binning joint contact-angle analyzer.

:class:`CoupledBinning3DAnalyzer` is the 3D extension of the joint
binning fit (:class:`CoupledBinning2DAnalyzer`). Instead of projecting
atoms onto a 2D ``(xi, zi)`` plane and exploiting radial symmetry, it
bins the full 3D density ``rho(xi, yi, zi)`` and fits a nine-parameter
hyperbolic-tangent model (``rho1, rho2, R_eq, xi_c, yi_c, zi_c, zi_0,
t1, t2``) directly.

Use it when:

- the droplet is spherical AND you want to avoid the radial-symmetry
  assumption baked into the 2D fit (e.g. you suspect asymmetry from
  an anisotropic wall or wetting heterogeneity);
- you have many frames per batch — a 3D density grid needs more
  sampling than a 2D one to reach the same per-cell noise.

Cylindrical droplets are rejected at construction: their translational
symmetry along the cylinder axis means the 3D fit reduces to the 2D
fit already implemented by :class:`CoupledBinning2DAnalyzer`.

The algorithm body (3D histogram → joint tanh fit) is stubbed with
``NotImplementedError`` at this skeleton stage. The worker-pool wiring
is real, so misconfigurations and per-batch exception handling are
exercised end-to-end.
"""

import logging
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis.base import (
    _BatchedTrajectoryAnalyzer,
    build_parser,
    gather_batch_coords,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledBinning3DBatchResult,
    CoupledBinning3DResults,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator

logger = logging.getLogger(__name__)


class CoupledBinning3DAnalyzer(_BatchedTrajectoryAnalyzer):
    """Joint contact-angle fit on a 3D binned density grid.

    Parameters
    ----------
    parser : BaseParser
        Trajectory parser. Only ``parser.filepath`` and
        ``parser.frame_count()`` are read in the parent process; each
        worker rebuilds its own parser.
    atom_indices : ndarray, optional
        Indices of the liquid atoms.
    droplet_geometry : DropletGeometry or str, default ``"spherical"``
        Must be spherical. Cylindrical droplets are rejected at
        construction because their translational symmetry already
        collapses the 3D problem onto the 2D one solved by
        :class:`CoupledBinning2DAnalyzer`.
    binning_params : dict, optional
        3D grid spec with keys ``"xi_0"``, ``"xi_f"``, ``"nbins_xi"``,
        ``"yi_0"``, ``"yi_f"``, ``"nbins_yi"``, ``"zi_0"``, ``"zi_f"``,
        ``"nbins_zi"``. If ``None``, a heuristic default is used (a
        third of the largest in-plane box dimension; 50 × 50 × 50
        bins). The heuristic is rarely optimal and emits a warning.
    initial_params : list[float], optional
        Initial guess for the nine tanh-model parameters
        ``[rho1, rho2, R_eq, xi_c, yi_c, zi_c, zi_0, t1, t2]``.
        Defaults to values consistent with the 2D model's defaults
        plus ``xi_c=0, yi_c=0`` (assuming the droplet is recentered).
    temporal_aggregator : TemporalAggregator, optional
        Defaults to a single fully pooled batch
        (``batch_size=-1``). The 3D density needs more frames than the
        2D one for comparable per-cell noise.
    precentered : bool, default ``False``
        Skip per-frame circular-mean PBC recentering.
    """

    #: Per-process worker state — shadowed from the parent so this
    #: subclass writes to its own slot.
    _WORKER_STATE: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        parser: Any,
        atom_indices: np.ndarray | None = None,
        droplet_geometry: DropletGeometry | str = "spherical",
        *,
        binning_params: dict[str, Any] | None = None,
        initial_params: list[float] | None = None,
        temporal_aggregator: TemporalAggregator | None = None,
        precentered: bool = False,
    ) -> None:
        super().__init__(
            parser=parser,
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            temporal_aggregator=temporal_aggregator
            or TemporalAggregator(batch_size=-1),
            precentered=precentered,
        )
        if not self.droplet_geometry.is_spherical:
            raise ValueError(
                "CoupledBinning3DAnalyzer only supports spherical droplets; "
                f"got droplet_geometry={self.droplet_geometry.name!r}. "
                "For cylindrical droplets use CoupledBinning2DAnalyzer — "
                "the 3D fit collapses onto the 2D one by translational "
                "symmetry along the cylinder axis."
            )
        self.binning_params = binning_params
        self.initial_params = initial_params

    # ------------------------------------------------------------------
    # _BatchedTrajectoryAnalyzer extension points.
    # ------------------------------------------------------------------

    def _tqdm_desc(self) -> str:
        return "CoupledBinning3DAnalyzer (spherical)"

    def _init_args(self) -> tuple:
        return (
            self.parser.filepath,
            self.atom_indices,
            self.droplet_geometry,
            self.binning_params,
            self.initial_params,
            self.precentered,
        )

    @staticmethod
    def _init_worker(
        filename: str,
        atom_indices: np.ndarray,
        droplet_geometry: DropletGeometry,
        binning_params: dict[str, Any] | None,
        initial_params: list[float] | None,
        precentered: bool,
    ) -> None:
        cls = CoupledBinning3DAnalyzer
        cls._WORKER_STATE.clear()
        cls._WORKER_STATE.update(
            parser=build_parser(filename),
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            binning_params=binning_params,
            initial_params=initial_params,
            precentered=precentered,
        )

    @staticmethod
    def _process_batch_worker(
        frame_indices: list[int],
    ) -> CoupledBinning3DBatchResult | None:
        state = CoupledBinning3DAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        precentered: bool = state["precentered"]
        try:
            # Pooled liquid-atom coordinates across the batch. The
            # CoupledBinning3D algorithm then builds the 3D histogram
            # and fits the nine-parameter tanh model — both stubbed.
            coords, center, max_dist = gather_batch_coords(
                parser=parser,
                frame_indices=frame_indices,
                atom_indices=atom_indices,
                droplet_geometry=droplet_geometry,
                precentered=precentered,
            )
            raise NotImplementedError(
                "coupled 3D-binning joint fit not implemented in skeleton."
            )
        except Exception as e:
            logger.error(f"Error processing batch {frame_indices}: {e}", exc_info=True)
            return None

    def _build_results(
        self, batches: list[CoupledBinning3DBatchResult]
    ) -> CoupledBinning3DResults:
        return CoupledBinning3DResults(
            batches=batches,
            method_metadata={
                "droplet_geometry": self.droplet_geometry.name,
                "binning_params": self.binning_params,
                "initial_params": self.initial_params,
                "batch_size": self.temporal_aggregator.batch_size,
            },
        )
