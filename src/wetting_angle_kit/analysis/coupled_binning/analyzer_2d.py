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
"""

import logging
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis._base import (
    _BatchedTrajectoryAnalyzer,
    build_parser,
)
from wetting_angle_kit.analysis.coupled_binning._models import (
    _PARAM_NAMES,
    _default_binning_params,
    _HyperbolicTangentModel2D,
    edges_from_bin_width,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledBinning2DBatchResult,
    CoupledBinning2DResults,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.io_utils import project_to_profile

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
        2D grid spec with keys ``"xi_0"``, ``"xi_f"``, ``"bin_width_x"``,
        ``"zi_0"``, ``"zi_f"``, ``"bin_width_z"``. The range bounds
        are honoured exactly; the effective cell width is rounded to
        fit and may differ slightly from the requested ``bin_width_*``.
        If ``None``, an atom-derived default is used: ``xi/zi`` span
        half the largest in-plane box dimension, with ``bin_width =
        0.5 Å`` (half the model's default interface thickness ``t1``).
        A warning is emitted when the default is used.
    initial_params : list[float], optional
        Initial guess for the seven tanh-model parameters
        ``[rho1, rho2, R_eq, zi_c, zi_0, t1, t2]``. Defaults to the
        values tuned for room-temperature water in the existing
        :class:`_HyperbolicTangentModel2D`.
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
        if binning_params is None:
            binning_params = _default_binning_params(parser)
        self.binning_params = binning_params
        self.initial_params = initial_params
        # Cylinder dV normalisation needs the box length along the
        # cylinder axis; read it once at construction (per legacy).
        self.box_dimension: float | None
        if self.droplet_geometry.is_cylinder:
            if self.droplet_geometry.cylinder_axis == "x":
                self.box_dimension = float(parser.box_size_x(frame_index=0))
            else:
                self.box_dimension = float(parser.box_size_y(frame_index=0))
        else:
            self.box_dimension = None

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
            self.box_dimension,
        )

    @staticmethod
    def _init_worker(
        filename: str,
        atom_indices: np.ndarray,
        droplet_geometry: DropletGeometry,
        binning_params: dict[str, Any],
        initial_params: list[float] | None,
        precentered: bool,
        box_dimension: float | None,
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
            box_dimension=box_dimension,
        )

    @staticmethod
    def _process_batch_worker(
        frame_indices: list[int],
    ) -> CoupledBinning2DBatchResult | None:
        state = CoupledBinning2DAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        binning_params: dict[str, Any] = state["binning_params"]
        initial_params: list[float] | None = state["initial_params"]
        precentered: bool = state["precentered"]
        box_dimension: float | None = state["box_dimension"]
        # Per-frame progress callback (inline mode only); see
        # :meth:`_BatchedTrajectoryAnalyzer._run_inline`.
        progress_callback = state.get("progress_callback")
        try:
            # Per-frame ``(xi, zi)`` projection, matching the legacy
            # ``BinningBatchFitter.get_profile_coordinates`` so the
            # joint fit sees the same projected coordinates.
            r_chunks: list[np.ndarray] = []
            z_chunks: list[np.ndarray] = []
            for frame_idx in frame_indices:
                positions = parser.parse(frame_index=frame_idx, indices=atom_indices)
                box_size: tuple[float, float] | None = None
                if not precentered:
                    box_size = (
                        parser.box_size_x(frame_index=frame_idx),
                        parser.box_size_y(frame_index=frame_idx),
                    )
                r_frame, z_frame = project_to_profile(
                    positions, droplet_geometry.name, box_size=box_size
                )
                r_chunks.append(r_frame)
                z_chunks.append(z_frame)
                if progress_callback is not None:
                    progress_callback(1)
            r_values = np.concatenate(r_chunks) if r_chunks else np.empty(0)
            z_values = np.concatenate(z_chunks) if z_chunks else np.empty(0)
            n_frames = len(frame_indices)

            # Build the 2D density grid + apply geometry-aware dV
            # normalisation. Cell width comes from binning_params;
            # the range bounds are honoured exactly and the effective
            # cell width may differ by a few percent.
            xi_edges = edges_from_bin_width(
                binning_params["xi_0"],
                binning_params["xi_f"],
                binning_params["bin_width_x"],
            )
            zi_edges = edges_from_bin_width(
                binning_params["zi_0"],
                binning_params["zi_f"],
                binning_params["bin_width_z"],
            )
            counts, _, _ = np.histogram2d(r_values, z_values, bins=(xi_edges, zi_edges))
            dxi = float(xi_edges[1] - xi_edges[0])
            dzi = float(zi_edges[1] - zi_edges[0])
            xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
            zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
            if droplet_geometry.is_cylinder:
                assert box_dimension is not None
                dV = 2.0 * box_dimension * dxi * dzi
                rho_cc = counts / dV
            else:
                dV_per_row = 2.0 * np.pi * xi_cc * dxi * dzi
                rho_cc = counts / dV_per_row[:, np.newaxis]
            if n_frames > 0:
                rho_cc /= n_frames

            # Joint tanh fit. ``_HyperbolicTangentModel2D`` expects the
            # density and grid axes flattened in Fortran order — same
            # as the legacy ``BinningBatchFitter.process_batch``.
            model = _HyperbolicTangentModel2D(initial_params=initial_params)
            msh_zi_grid, msh_xi_grid = np.meshgrid(zi_cc, xi_cc)
            n_flat = len(xi_cc) * len(zi_cc)
            msh_zi = msh_zi_grid.reshape(n_flat, order="F")
            msh_xi = msh_xi_grid.reshape(n_flat, order="F")
            msh_rho = rho_cc.reshape(n_flat, order="F")
            model.fit((msh_xi, msh_zi), msh_rho)
            angle = float(model.compute_contact_angle())
            params = model.params
            if params is None:
                raise RuntimeError(
                    "_HyperbolicTangentModel2D did not set model parameters; "
                    "cannot build CoupledBinning2DBatchResult."
                )
            model_params = {
                name: float(value)
                for name, value in zip(_PARAM_NAMES, params, strict=False)
            }
            return CoupledBinning2DBatchResult(
                frames=list(frame_indices),
                angle=angle,
                model_params=model_params,
                xi_grid=xi_cc.copy(),
                zi_grid=zi_cc.copy(),
                density=rho_cc,
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
