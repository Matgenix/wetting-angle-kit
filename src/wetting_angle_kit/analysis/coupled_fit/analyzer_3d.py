"""Coupled 3D-binning joint contact-angle analyzer.

:class:`CoupledFit3DAnalyzer` is the 3D extension of the joint
binning fit (:class:`CoupledFit2DAnalyzer`). Instead of projecting
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
fit already implemented by :class:`CoupledFit2DAnalyzer`.
"""

import logging
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis._base import (
    build_parser,
    gather_batch_coords,
)
from wetting_angle_kit.analysis._grid_utils import edges_from_bin_width
from wetting_angle_kit.analysis.coupled_fit._base import (
    _CoupledFitAnalyzer,
    fit_model_params,
)
from wetting_angle_kit.analysis.coupled_fit._models import (
    _default_binning_params_3d,
    _HyperbolicTangentModel3D,
)
from wetting_angle_kit.analysis.density_estimator import (
    DensityEstimator,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledFit3DBatchResult,
    CoupledFit3DResults,
)

logger = logging.getLogger(__name__)


class CoupledFit3DAnalyzer(_CoupledFitAnalyzer):
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
        :class:`CoupledFit2DAnalyzer`.
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

    #: Results dataclass produced by the shared ``_build_results``.
    _RESULTS_CLS: ClassVar[type] = CoupledFit3DResults

    def _check_geometry(self) -> None:
        if not self.droplet_geometry.is_spherical:
            raise ValueError(
                "CoupledFit3DAnalyzer only supports spherical droplets; "
                f"got droplet_geometry={self.droplet_geometry.name!r}. "
                "For cylindrical droplets use CoupledFit2DAnalyzer — "
                "the 3D fit collapses onto the 2D one by translational "
                "symmetry along the cylinder axis."
            )

    def _default_binning_params(self, parser: Any) -> dict[str, Any]:
        return _default_binning_params_3d(parser)

    # ------------------------------------------------------------------
    # _BatchedTrajectoryAnalyzer extension points.
    # ------------------------------------------------------------------

    def _init_args(self) -> tuple:
        return (
            self.parser.filepath,
            self.atom_indices,
            self.droplet_geometry,
            self.binning_params,
            self.density_estimator,
            self.initial_params,
            self.precentered,
        )

    @staticmethod
    def _init_worker(
        filename: str,
        atom_indices: np.ndarray,
        droplet_geometry: DropletGeometry,
        binning_params: dict[str, Any],
        density_estimator: DensityEstimator,
        initial_params: list[float] | None,
        precentered: bool,
    ) -> None:
        cls = CoupledFit3DAnalyzer
        cls._WORKER_STATE.clear()
        cls._WORKER_STATE.update(
            parser=build_parser(filename),
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            binning_params=binning_params,
            density_estimator=density_estimator,
            initial_params=initial_params,
            precentered=precentered,
        )

    @staticmethod
    def _process_batch_worker(
        frame_indices: list[int],
    ) -> CoupledFit3DBatchResult | None:
        state = CoupledFit3DAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        binning_params: dict[str, Any] = state["binning_params"]
        density_estimator: DensityEstimator = state["density_estimator"]
        initial_params: list[float] | None = state["initial_params"]
        precentered: bool = state["precentered"]
        # Per-frame progress callback (inline mode only); see
        # :meth:`_BatchedTrajectoryAnalyzer._run_inline`.
        progress_callback = state.get("progress_callback")
        try:
            # Per-frame PBC recentering, then drop each frame's atoms
            # in the droplet-centred ``(x, y)`` frame (z stays in the
            # lab frame so the wall position retains physical meaning).
            coords, _ = gather_batch_coords(
                parser=parser,
                frame_indices=frame_indices,
                atom_indices=atom_indices,
                droplet_geometry=droplet_geometry,
                precentered=precentered,
                center_on_com=True,
                progress_callback=progress_callback,
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
            rho = density_estimator.evaluate_3d(
                atoms_pooled=coords,
                n_frames=n_frames,
                droplet_geometry=droplet_geometry,
                xi_edges=xi_edges,
                yi_edges=yi_edges,
                zi_edges=zi_edges,
            )

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
            angle, model_params = fit_model_params(
                model, (xi_flat, yi_flat, zi_flat), rho_flat
            )
            return CoupledFit3DBatchResult(
                frames=list(frame_indices),
                angle=angle,
                model_params=model_params,
                xi_grid=xi_cc.copy(),
                yi_grid=yi_cc.copy(),
                zi_grid=zi_cc.copy(),
                density=rho,
            )
        except Exception as e:
            logger.error(f"Error processing batch {frame_indices}: {e}", exc_info=True)
            return None
