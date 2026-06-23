"""Shared scaffolding for the coupled-fit analyzers.

:class:`_CoupledFitAnalyzer` factors out everything the 2D and 3D
coupled-fit analyzers share — the constructor, the progress-bar label,
the results packaging, and the model fit / parameter extraction —
leaving each concrete analyzer to supply only its dimensionality-specific
per-batch density binning and tanh-model wiring (the worker triple).
"""

import logging
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis._base import _BatchedTrajectoryAnalyzer
from wetting_angle_kit.analysis.coupled_fit._models import _HyperbolicTangentModel
from wetting_angle_kit.analysis.density_estimator import DensityEstimator
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.temporal import TemporalAggregator

logger = logging.getLogger(__name__)


def fit_model_params(
    model: _HyperbolicTangentModel,
    x_data: tuple[np.ndarray, ...],
    density_flat: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Fit ``model`` and return ``(contact_angle, model_params dict)``.

    Shared by the 2D and 3D workers: the only dimensionality-specific
    inputs are the already-flattened ``x_data`` / ``density_flat`` the
    caller assembles from its grid. The parameter names come from the
    model itself (:attr:`_HyperbolicTangentModel.param_names`), so the
    same helper serves both the 7- and 9-parameter layouts.
    """
    model.fit(x_data, density_flat)
    angle = float(model.compute_contact_angle())
    params = model.params
    if params is None:
        raise RuntimeError(
            f"{type(model).__name__} did not set model parameters; "
            "cannot build a coupled-fit batch result."
        )
    model_params = {
        name: float(value)
        for name, value in zip(model.param_names, params, strict=False)
    }
    return angle, model_params


class _CoupledFitAnalyzer(_BatchedTrajectoryAnalyzer):
    """Shared base for the two coupled-fit analyzers.

    Concrete subclasses (:class:`CoupledFit2DAnalyzer`,
    :class:`CoupledFit3DAnalyzer`) provide:

    - ``_RESULTS_CLS`` — the results dataclass to wrap the batches in;
    - ``_default_grid_params(parser)`` — the atom-derived default
      grid used when ``grid_params`` is ``None``;
    - the worker triple (``_init_args`` / ``_init_worker`` /
      ``_process_batch_worker``), which differs by grid dimensionality;
    - optionally ``_check_geometry`` (the 3D fit rejects cylinders) and
      ``_post_init`` (the 2D fit reads the cylinder box length).
    """

    #: Results dataclass produced by :meth:`_build_results`.
    _RESULTS_CLS: ClassVar[type]

    def __init__(
        self,
        parser: Any,
        atom_indices: np.ndarray | None = None,
        droplet_geometry: DropletGeometry | str = "spherical",
        *,
        grid_params: dict[str, Any] | None = None,
        density_estimator: DensityEstimator | None = None,
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
        # Reject unsupported geometries before the (warning-emitting)
        # default-grid derivation runs.
        self._check_geometry()
        if grid_params is None:
            grid_params = self._default_grid_params(parser)
        self.grid_params = grid_params
        self.density_estimator = density_estimator or DensityEstimator.binning()
        self.initial_params = initial_params
        self._post_init(parser)

    # ------------------------------------------------------------------
    # Subclass hooks.
    # ------------------------------------------------------------------

    def _check_geometry(self) -> None:
        """Reject unsupported droplet geometries. Default: accept all."""

    @abstractmethod
    def _default_grid_params(self, parser: Any) -> dict[str, Any]:
        """Atom-derived default grid spec when ``grid_params`` is None."""

    def _post_init(self, parser: Any) -> None:
        """Construction hook run after the common fields are set."""

    # ------------------------------------------------------------------
    # Shared _BatchedTrajectoryAnalyzer extension points.
    # ------------------------------------------------------------------

    def _tqdm_desc(self) -> str:
        return (
            f"{type(self).__name__} "
            f"({self.droplet_geometry.name} / {self.density_estimator.kind})"
        )

    def _build_results(self, batches: list[Any]) -> Any:
        return self._RESULTS_CLS(
            batches=batches,
            method_metadata={
                "droplet_geometry": self.droplet_geometry.name,
                "grid_params": self.grid_params,
                "initial_params": self.initial_params,
                "batch_size": self.temporal_aggregator.batch_size,
            },
        )
