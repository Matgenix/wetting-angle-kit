"""Coupled 2D-binning joint contact-angle analyzer.

:class:`CoupledFit2DAnalyzer` is the modern incarnation of the
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
:class:`CoupledFit3DAnalyzer`.
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
    _default_binning_params as _default_binning_params_2d,
)
from wetting_angle_kit.analysis.coupled_fit._models import (
    _HyperbolicTangentModel2D,
)
from wetting_angle_kit.analysis.density_estimator import (
    DensityEstimator,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit2DResults,
)

logger = logging.getLogger(__name__)


class CoupledFit2DAnalyzer(_CoupledFitAnalyzer):
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
    density_estimator : DensityEstimator, optional
        How the per-cell density is computed from the pooled atom
        positions. Built via :meth:`DensityEstimator.binning` (the
        default, top-hat histogram with geometry-aware ``dV``
        normalisation) or :meth:`DensityEstimator.gaussian`
        (3D Gaussian KDE evaluated at the cell centres; same kernel
        the ``rays`` / ``grid`` with the Gaussian extractors use).
        Switching to the Gaussian variant smooths out per-cell
        Poisson noise — useful on per-frame / small-batch analyses
        where the histogram density is degenerate.
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

    #: Results dataclass produced by the shared ``_build_results``.
    _RESULTS_CLS: ClassVar[type] = CoupledFit2DResults

    def _default_binning_params(self, parser: Any) -> dict[str, Any]:
        return _default_binning_params_2d(parser)

    def _post_init(self, parser: Any) -> None:
        # Cylinder dV normalisation needs the box length along the
        # cylinder axis; read it once at construction.
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

    def _init_args(self) -> tuple:
        return (
            self.parser.filepath,
            self.atom_indices,
            self.droplet_geometry,
            self.binning_params,
            self.density_estimator,
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
        density_estimator: DensityEstimator,
        initial_params: list[float] | None,
        precentered: bool,
        box_dimension: float | None,
    ) -> None:
        cls = CoupledFit2DAnalyzer
        cls._WORKER_STATE.clear()
        cls._WORKER_STATE.update(
            parser=build_parser(filename),
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            binning_params=binning_params,
            density_estimator=density_estimator,
            initial_params=initial_params,
            precentered=precentered,
            box_dimension=box_dimension,
        )

    @staticmethod
    def _process_batch_worker(
        frame_indices: list[int],
    ) -> CoupledFit2DBatchResult | None:
        state = CoupledFit2DAnalyzer._WORKER_STATE
        parser = state["parser"]
        atom_indices: np.ndarray = state["atom_indices"]
        droplet_geometry: DropletGeometry = state["droplet_geometry"]
        binning_params: dict[str, Any] = state["binning_params"]
        density_estimator: DensityEstimator = state["density_estimator"]
        initial_params: list[float] | None = state["initial_params"]
        precentered: bool = state["precentered"]
        box_dimension: float | None = state["box_dimension"]
        # Per-frame progress callback (inline mode only); see
        # :meth:`_BatchedTrajectoryAnalyzer._run_inline`.
        progress_callback = state.get("progress_callback")
        try:
            # Per-frame PBC recentering + droplet-centring in (x, y);
            # ``z`` stays in the lab frame so wall position retains
            # physical meaning. The pooled 3D positions are then
            # handed to the density estimator strategy, which picks
            # its own projection (radial for spherical, |x| for
            # cylinder) and density rule (histogram vs Gaussian KDE).
            atoms_pooled, _ = gather_batch_coords(
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
            zi_edges = edges_from_bin_width(
                binning_params["zi_0"],
                binning_params["zi_f"],
                binning_params["bin_width_z"],
            )
            xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
            zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
            rho_cc = density_estimator.evaluate_2d(
                atoms_pooled=atoms_pooled,
                n_frames=n_frames,
                droplet_geometry=droplet_geometry,
                xi_edges=xi_edges,
                zi_edges=zi_edges,
                box_dimension=box_dimension,
            )

            # Joint tanh fit. ``_HyperbolicTangentModel2D`` expects the
            # density and grid axes flattened in Fortran order so the
            # ``(xi, zi)`` pairs line up with their density values.
            model = _HyperbolicTangentModel2D(initial_params=initial_params)
            msh_zi_grid, msh_xi_grid = np.meshgrid(zi_cc, xi_cc)
            n_flat = len(xi_cc) * len(zi_cc)
            msh_zi = msh_zi_grid.reshape(n_flat, order="F")
            msh_xi = msh_xi_grid.reshape(n_flat, order="F")
            msh_rho = rho_cc.reshape(n_flat, order="F")
            angle, model_params = fit_model_params(model, (msh_xi, msh_zi), msh_rho)
            return CoupledFit2DBatchResult(
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
