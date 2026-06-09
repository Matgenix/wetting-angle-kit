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
import warnings
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import curve_fit

from wetting_angle_kit.analysis.base import (
    _BatchedTrajectoryAnalyzer,
    build_parser,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    CoupledBinning3DBatchResult,
    CoupledBinning3DResults,
)
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.io_utils import recenter_droplet_pbc

logger = logging.getLogger(__name__)

_PARAM_NAMES_3D = (
    "rho1",
    "rho2",
    "R_eq",
    "xi_c",
    "yi_c",
    "zi_c",
    "zi_0",
    "t1",
    "t2",
)


class _HyperbolicTangentModel3D:
    """3D extension of the binning method's hyperbolic-tangent model.

    Density factorises into a radial sigmoid centred at ``(xi_c, yi_c,
    zi_c)`` and a vertical sigmoid above the wall ``zi_0``:

    ::

        rho(xi, yi, zi) = g(r) * h(zi - zi_0),
            g(r) = 0.5 * [(rho1 + rho2) - (rho1 - rho2) * tanh(2 (r - R_eq) / t1)],
            h(z) = 0.5 * [1 + tanh(2 z / t2)],
            r    = sqrt((xi - xi_c)^2 + (yi - yi_c)^2 + (zi - zi_c)^2).

    Bounds keep densities and lengths in their physical ranges, same as
    the 2D model. ``xi_c`` / ``yi_c`` carry the only extra degrees of
    freedom over the 2D fit.
    """

    #: Initial guess tuned for room-temperature water; the two
    #: horizontal centres default to ``0`` because the analyzer
    #: pre-centers the atoms on the droplet COM before binning.
    DEFAULT_INITIAL_PARAMS = (1e-3, 3e-2, 40.0, 0.0, 0.0, 20.0, 4.0, 1.0, 1.0)

    # Bounds vector order matches DEFAULT_INITIAL_PARAMS.
    _PARAM_LOWER = np.array(
        [0.0, 0.0, 1e-6, -np.inf, -np.inf, -np.inf, -np.inf, 1e-6, 1e-6]
    )
    _PARAM_UPPER = np.array([np.inf] * 9)

    def __init__(self, initial_params: list[float] | None = None) -> None:
        if initial_params is None:
            initial_params = list(self.DEFAULT_INITIAL_PARAMS)
        self.params: list[float] | np.ndarray | None = initial_params
        self.covariance: np.ndarray | None = None

    @staticmethod
    def _fitting_function(
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        rho1: float,
        rho2: float,
        R_eq: float,
        xi_c: float,
        yi_c: float,
        zi_c: float,
        zi_0: float,
        t1: float,
        t2: float,
    ) -> np.ndarray:
        xi, yi, zi = x[0], x[1], x[2]
        r = np.sqrt((xi - xi_c) ** 2 + (yi - yi_c) ** 2 + (zi - zi_c) ** 2)
        g_r = 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
        h_z = 0.5 * (1.0 + np.tanh(2 * (zi - zi_0) / t2))
        return g_r * h_z

    def fit(
        self,
        x_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        density_data: np.ndarray,
    ) -> "_HyperbolicTangentModel3D":
        self.params, self.covariance = curve_fit(
            self._fitting_function,
            x_data,
            density_data,
            p0=self.params,
            bounds=(self._PARAM_LOWER, self._PARAM_UPPER),
            maxfev=1_000_000,
        )
        self._warn_if_at_bounds()
        return self

    def _warn_if_at_bounds(self) -> None:
        if self.params is None:
            return
        tol = 1e-6
        at_bound = []
        for name, value, lo, hi in zip(
            _PARAM_NAMES_3D,
            self.params,
            self._PARAM_LOWER,
            self._PARAM_UPPER,
            strict=False,
        ):
            if np.isfinite(lo) and abs(value - lo) < tol * max(1.0, abs(lo)):
                at_bound.append(f"{name}={value:.3g} at lower bound {lo}")
            elif np.isfinite(hi) and abs(value - hi) < tol * max(1.0, abs(hi)):
                at_bound.append(f"{name}={value:.3g} at upper bound {hi}")
        if at_bound:
            warnings.warn(
                "3D hyperbolic tangent fit converged with parameter(s) at "
                "the physical bound, suggesting a poor fit: " + "; ".join(at_bound),
                RuntimeWarning,
                stacklevel=3,
            )

    def compute_contact_angle(self) -> float:
        """Return the contact angle (degrees) implied by the fitted parameters.

        Same geometric formula as the 2D model: the sphere of radius
        ``R_eq`` centred at ``(xi_c, yi_c, zi_c)`` intersects the wall
        plane ``z = zi_0`` in a circle whose tangent makes the contact
        angle with the wall.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle.")
        R_eq = float(self.params[2])
        zi_c = float(self.params[5])
        zi_0 = float(self.params[6])
        discriminant = R_eq**2 - (zi_0 - zi_c) ** 2
        if discriminant < 0:
            warnings.warn(
                "3D fit wall is outside the fitted droplet sphere "
                f"(R_eq={R_eq:.3f}, |zi_0 - zi_c|="
                f"{abs(zi_0 - zi_c):.3f}); contact angle is undefined.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        xi_cross = np.sqrt(discriminant)
        return float((np.pi / 2 - np.arctan((zi_0 - zi_c) / xi_cross)) * 180 / np.pi)


def _heuristic_binning_params_3d(parser: Any) -> dict[str, Any]:
    """Build a heuristic 3D binning grid centred on the droplet COM.

    Same one-third-of-box heuristic as the 2D version but tripled
    along all three axes. Emits a warning because the user almost
    always wants to override this.
    """
    half = int(
        np.max(
            np.array(
                [
                    parser.box_size_y(frame_index=0),
                    parser.box_size_x(frame_index=0),
                ]
            )
        )
        / 6
    )
    warnings.warn(
        "binning_params was not supplied; using a heuristic default "
        f"(xi/yi in [-{half}, {half}], zi in [0, {2 * half}], 30^3 bins). "
        "For accurate density fields, supply system-specific "
        "binning_params matching your droplet size and per-frame sampling.",
        UserWarning,
        stacklevel=3,
    )
    return {
        "xi_0": -half,
        "xi_f": half,
        "nbins_xi": 30,
        "yi_0": -half,
        "yi_f": half,
        "nbins_yi": 30,
        "zi_0": 0.0,
        "zi_f": 2 * half,
        "nbins_zi": 30,
    }


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
        ``"nbins_zi"``. ``xi``/``yi`` are in the droplet-centred frame
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
            binning_params = _heuristic_binning_params_3d(parser)
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
            coords = (
                np.concatenate(coord_chunks, axis=0)
                if coord_chunks
                else np.empty((0, 3))
            )
            n_frames = len(frame_indices)

            xi_edges = np.linspace(
                binning_params["xi_0"],
                binning_params["xi_f"],
                int(binning_params["nbins_xi"]),
            )
            yi_edges = np.linspace(
                binning_params["yi_0"],
                binning_params["yi_f"],
                int(binning_params["nbins_yi"]),
            )
            zi_edges = np.linspace(
                binning_params["zi_0"],
                binning_params["zi_f"],
                int(binning_params["nbins_zi"]),
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
