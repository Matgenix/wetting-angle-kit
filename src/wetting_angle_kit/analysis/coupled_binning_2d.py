"""Coupled 2D-binning joint contact-angle analyzer.

:class:`CoupledBinning2DAnalyzer` is the modern incarnation of the
package's original binning method. Unlike :class:`TrajectoryAnalyzer`
it does not separate interface extraction, wall detection, and surface
fit — a seven-parameter hyperbolic-tangent model (rho1, rho2, R_eq,
zi_c, zi_0, t1, t2) solves all three jointly on a binned 2D density
grid.

Use it when:

- the droplet is in the spherical-cap regime (cylindrical works too;
  the 2D fit exploits the cylinder's translational symmetry);
- you have many frames per batch so the binned density is
  well-sampled;
- you want a single robust estimate per batch and don't need per-frame
  time resolution.

For per-frame analysis with separable strategies use
:class:`TrajectoryAnalyzer` instead. For the 3D extension of this
analyzer (relaxing the radial symmetry assumption) see
:class:`CoupledBinning3DAnalyzer`.

The algorithm body (projection → 2D histogram → tanh fit) is stubbed
with ``NotImplementedError`` at this skeleton stage. The
worker-pool wiring is real, so misconfigurations and per-batch
exception handling are exercised end-to-end.
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
    CoupledBinning2DBatchResult,
    CoupledBinning2DResults,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator

logger = logging.getLogger(__name__)


class CoupledBinning2DAnalyzer(_BatchedTrajectoryAnalyzer):
    """Joint contact-angle fit on a 2D binned density grid.

    Parameters
    ----------
    parser : BaseParser
        Trajectory parser. Only ``parser.filepath`` and
        ``parser.frame_count()`` are read in the parent process; each
        worker rebuilds its own parser.
    atom_indices : ndarray, optional
        Indices of the liquid atoms.
    droplet_geometry : DropletGeometry or str, default ``"spherical"``
        Either an instance or the bare name string. Determines the
        per-frame projection onto the ``(xi, zi)`` plane: spherical
        droplets use the in-plane radial coordinate
        ``xi = sqrt(x^2 + y^2)``; cylindrical droplets use the
        coordinate perpendicular to the cylinder axis.
    binning_params : dict, optional
        2D grid spec with keys ``"xi_0"``, ``"xi_f"``, ``"nbins_xi"``,
        ``"zi_0"``, ``"zi_f"``, ``"nbins_zi"``. If ``None``, a
        heuristic default is used (a third of the largest in-plane box
        dimension; 50 × 50 bins). The heuristic is rarely optimal for
        a specific system and emits a warning when used.
    initial_params : list[float], optional
        Initial guess for the seven tanh-model parameters
        ``[rho1, rho2, R_eq, zi_c, zi_0, t1, t2]``. Defaults to the
        values tuned for room-temperature water in the existing
        ``HyperbolicTangentModel``.
    temporal_aggregator : TemporalAggregator, optional
        Defaults to a single fully pooled batch
        (``batch_size=-1``) — the coupled fit benefits from as much
        statistics as possible. Set ``batch_size=N`` to compute
        independent angles for each ``N``-frame block.
    precentered : bool, default ``False``
        Skip per-frame circular-mean PBC recentering. Setting this on
        a trajectory that does NOT satisfy the precondition will
        produce wrong results.
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
        self.binning_params = binning_params
        self.initial_params = initial_params

    # ------------------------------------------------------------------
    # _BatchedTrajectoryAnalyzer extension points.
    # ------------------------------------------------------------------

    def _tqdm_desc(self) -> str:
        return f"CoupledBinning2DAnalyzer ({self.droplet_geometry.name})"

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
        cls = CoupledBinning2DAnalyzer
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
    ) -> CoupledBinning2DBatchResult | None:
        state = CoupledBinning2DAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        precentered: bool = state["precentered"]
        try:
            # Pooled liquid-atom coordinates across the batch. The
            # CoupledBinning2D algorithm then projects to (xi, zi),
            # builds the histogram, and fits the tanh model — all
            # currently stubbed.
            coords, center, max_dist = gather_batch_coords(
                parser=parser,
                frame_indices=frame_indices,
                atom_indices=atom_indices,
                droplet_geometry=droplet_geometry,
                precentered=precentered,
            )
            raise NotImplementedError(
                "coupled 2D-binning joint fit not implemented in skeleton."
            )
        except Exception as e:
            logger.error(f"Error processing batch {frame_indices}: {e}", exc_info=True)
            return None

    def _build_results(
        self, batches: list[CoupledBinning2DBatchResult]
    ) -> CoupledBinning2DResults:
        return CoupledBinning2DResults(
            batches=batches,
            method_metadata={
                "droplet_geometry": self.droplet_geometry.name,
                "binning_params": self.binning_params,
                "initial_params": self.initial_params,
                "batch_size": self.temporal_aggregator.batch_size,
            },
        )
