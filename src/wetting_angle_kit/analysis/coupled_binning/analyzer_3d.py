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
"""

import logging
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis._base import (
    _BatchedTrajectoryAnalyzer,
    build_parser,
)
from wetting_angle_kit.analysis.coupled_binning._models import (
    _PARAM_NAMES_3D,
    _default_binning_params_3d,
    _HyperbolicTangentModel3D,
    edges_from_bin_width,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledBinning3DBatchResult,
    CoupledBinning3DResults,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.io_utils import recenter_droplet_pbc

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
        3D grid spec with keys ``"xi_0"``, ``"xi_f"``, ``"bin_width_x"``,
        ``"yi_0"``, ``"yi_f"``, ``"bin_width_y"``, ``"zi_0"``, ``"zi_f"``,
        ``"bin_width_z"``. The range bounds are honoured exactly; the
        effective cell width is rounded to fit. If ``None``, an
        atom-derived default is used (lateral half-box for all axes,
        ``bin_width = 1 Å`` to keep the 9-parameter NLLS tractable).
        ``xi``/``yi`` are in the droplet-centred frame
        (atoms are recentred on the per-frame COM before binning); ``zi``
        is in the lab frame so the wall position retains physical
        meaning. If ``None``, a heuristic default is used.
    initial_params : list[float], optional
        Initial guess for the nine tanh-model parameters
        ``[rho1, rho2, R_eq, xi_c, yi_c, zi_c, zi_0, t1, t2]``.
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
        if binning_params is None:
            binning_params = _default_binning_params_3d(parser)
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
        binning_params: dict[str, Any],
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
        binning_params: dict[str, Any] = state["binning_params"]
        initial_params: list[float] | None = state["initial_params"]
        precentered: bool = state["precentered"]
        # Per-frame progress callback (inline mode only); see
        # :meth:`_BatchedTrajectoryAnalyzer._run_inline`.
        progress_callback = state.get("progress_callback")
        try:
            # Per-frame PBC recentering, then drop each frame's atoms
            # in the droplet-centred ``(x, y)`` frame (z stays in the
            # lab frame so the wall position retains physical meaning).
            coord_chunks: list[np.ndarray] = []
            for frame_idx in frame_indices:
                positions = parser.parse(frame_index=frame_idx, indices=atom_indices)
                if precentered:
                    com = np.mean(positions, axis=0)
                else:
                    box_xy = (
                        parser.box_size_x(frame_index=frame_idx),
                        parser.box_size_y(frame_index=frame_idx),
                    )
                    positions, com = recenter_droplet_pbc(
                        positions, droplet_geometry.name, box_size=box_xy
                    )
                positions_centered = positions - np.array([com[0], com[1], 0.0])
                coord_chunks.append(positions_centered)
                if progress_callback is not None:
                    progress_callback(1)
            coords = (
                np.concatenate(coord_chunks, axis=0)
                if coord_chunks
                else np.empty((0, 3))
            )
            n_frames = len(frame_indices)

            xi_edges = edges_from_bin_width(
                binning_params["xi_0"],
                binning_params["xi_f"],
                binning_params["bin_width_x"],
            )
            yi_edges = edges_from_bin_width(
                binning_params["yi_0"],
                binning_params["yi_f"],
                binning_params["bin_width_y"],
            )
            zi_edges = edges_from_bin_width(
                binning_params["zi_0"],
                binning_params["zi_f"],
                binning_params["bin_width_z"],
            )
            counts, _ = np.histogramdd(coords, bins=(xi_edges, yi_edges, zi_edges))
            dxi = float(xi_edges[1] - xi_edges[0])
            dyi = float(yi_edges[1] - yi_edges[0])
            dzi = float(zi_edges[1] - zi_edges[0])
            rho = counts / (dxi * dyi * dzi)
            if n_frames > 0:
                rho /= n_frames

            xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
            yi_cc = 0.5 * (yi_edges[:-1] + yi_edges[1:])
            zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])

            # Flatten the 3D grid for the curve fit. ``np.meshgrid``
            # with ``indexing="ij"`` matches ``histogramdd``'s axis
            # convention, so a plain ``ravel`` keeps positions aligned
            # with density values.
            XI, YI, ZI = np.meshgrid(xi_cc, yi_cc, zi_cc, indexing="ij")
            xi_flat = XI.ravel()
            yi_flat = YI.ravel()
            zi_flat = ZI.ravel()
            rho_flat = rho.ravel()

            model = _HyperbolicTangentModel3D(initial_params=initial_params)
            model.fit((xi_flat, yi_flat, zi_flat), rho_flat)
            angle = model.compute_contact_angle()
            params = model.params
            if params is None:
                raise RuntimeError(
                    "_HyperbolicTangentModel3D did not set parameters; "
                    "cannot build CoupledBinning3DBatchResult."
                )
            model_params = {
                name: float(value)
                for name, value in zip(_PARAM_NAMES_3D, params, strict=False)
            }
            return CoupledBinning3DBatchResult(
                frames=list(frame_indices),
                angle=float(angle),
                model_params=model_params,
                xi_grid=xi_cc.copy(),
                yi_grid=yi_cc.copy(),
                zi_grid=zi_cc.copy(),
                density=rho,
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
