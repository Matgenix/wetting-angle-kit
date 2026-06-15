"""In-memory containers for trajectory analyzer outputs.

The unified :class:`TrajectoryAnalyzer` returns :class:`TrajectoryResults`
holding one :class:`BatchResult` per batch produced by the
:class:`TemporalAggregator`. The specific :class:`BatchResult` subclass
depends on the :class:`SurfaceFitter` kind:

- slicing fitters → :class:`SlicingBatchResult`
- whole fitters   → :class:`WholeBatchResult`

The two joint-fit analyzers — :class:`CoupledFit2DAnalyzer` and
:class:`CoupledFit3DAnalyzer` — each return their own results type
(:class:`CoupledFit2DResults`, :class:`CoupledFit3DResults`).
They carry density grids plus joint-fit parameters and are therefore
not part of the :class:`TrajectoryResults` hierarchy.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ``eq=False`` is used throughout because per-batch payloads contain
# numpy arrays, on which the auto-generated ``__eq__`` would call
# element-wise ``==`` and raise in a boolean context. Equality between
# result objects isn't a use case the package needs.


@dataclass(frozen=True, eq=False, kw_only=True)
class BatchResult:
    """Common fields shared by all per-batch trajectory results.

    All fields are keyword-only so that subclasses can interleave their
    own fields with the parent's defaulted ones without ordering
    constraints.

    Attributes
    ----------
    frames : list[int]
        Frame indices pooled into this batch.
    angle : float
        Representative contact angle for the batch (degrees). For
        slicing fits this is the mean across slices; for whole fits
        it is the single fitted angle.
    z_wall : float
        Wall-plane z used by the surface fitter for this batch (Å).
    rms_residual : float
        Aggregate fit residual (Å). For slicing fits, an aggregate of
        the per-slice circle-fit residuals; for whole fits, the
        single sphere/cylinder-fit residual.
    angle_std : float, optional
        Within-batch standard deviation of the contact angle
        (degrees), describing the spread of the per-batch ``angle``.
        For slicing fits, the ``nanstd`` of ``per_slice_angles``
        (always populated). For whole fits, the bootstrap std when the
        fitter was constructed with ``bootstrap_samples > 0``,
        otherwise ``None``.
    """

    frames: list[int]
    angle: float
    z_wall: float
    rms_residual: float
    angle_std: float | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class SlicingBatchResult(BatchResult):
    """Per-batch result from a slicing-kind surface fitter.

    All per-slice arrays are full length (one entry per attempted
    slice) and index-aligned: a slice that produced no valid contact
    angle (empty, too few points, degenerate circle fit, or a circle
    that does not reach the wall) is marked ``nan`` rather than dropped,
    so attrition is visible and the slice index is preserved.

    Attributes
    ----------
    per_slice_angles : ndarray
        ``(n_slices_total,)`` array of per-slice contact angles
        (degrees), with ``nan`` for slices that produced no angle.
        :attr:`BatchResult.angle` is ``nanmean`` of this array and
        :attr:`BatchResult.angle_std` its ``nanstd``.
    slice_surfaces : list[ndarray]
        One ``(M_i, 2)`` array per slice of interface points in the
        slice ``(x, z)`` plane (kept for every slice, including those
        that produced no angle).
    slice_popts : ndarray
        ``(n_slices_total, 4)`` array of fitted circle parameters per
        slice; columns ``[xc, zc, R, z_wall]``. Rows for slices with no
        valid fit are ``nan``.
    n_slices_total : int
        Number of slices the extractor produced for this batch.
    n_slices_used : int
        Number of those slices that produced a valid contact angle
        (the count of non-``nan`` entries in ``per_slice_angles``).
        ``n_slices_used < n_slices_total`` signals per-slice attrition.
    """

    per_slice_angles: np.ndarray
    slice_surfaces: list[np.ndarray]
    slice_popts: np.ndarray
    n_slices_total: int
    n_slices_used: int


@dataclass(frozen=True, eq=False, kw_only=True)
class WholeBatchResult(BatchResult):
    """Per-batch result from a whole-kind surface fitter.

    Attributes
    ----------
    interface_shell : ndarray
        ``(N, 3)`` array of interface points used in the fit, in the
        internal ``(x, y, z)`` frame.
    popt : ndarray
        Fitted shape parameters extended by the wall plane. Spherical
        fitter: ``[xc, yc, zc, R, z_wall]``. Cylinder fitter:
        ``[xc, zc, R, z_wall]``.
    """

    interface_shell: np.ndarray
    popt: np.ndarray


@dataclass(frozen=True, eq=False)
class CoupledFit2DBatchResult:
    """Per-batch result from :class:`CoupledFit2DAnalyzer`.

    Attributes
    ----------
    frames : list[int]
        Frame indices pooled into this batch.
    angle : float
        Contact angle (degrees) implied by the joint 2D tanh-model fit.
    model_params : dict[str, float]
        Fitted parameters of the 2D hyperbolic tangent model; keys are
        ``"rho1"``, ``"rho2"``, ``"R_eq"``, ``"zi_c"``, ``"zi_0"``,
        ``"t1"``, ``"t2"``.
    xi_grid : ndarray
        In-plane binning grid centers (Å).
    zi_grid : ndarray
        Vertical binning grid centers (Å).
    density : ndarray
        ``(len(xi_grid), len(zi_grid))`` binned density on the grid.
    """

    frames: list[int]
    angle: float
    model_params: dict[str, float]
    xi_grid: np.ndarray
    zi_grid: np.ndarray
    density: np.ndarray


@dataclass(frozen=True, eq=False)
class CoupledFit3DBatchResult:
    """Per-batch result from :class:`CoupledFit3DAnalyzer`.

    Only meaningful for spherical droplets; cylindrical droplets carry
    a translational symmetry along the cylinder axis that the 2D
    analyzer already exploits, so :class:`CoupledFit3DAnalyzer`
    rejects non-spherical geometries at construction.

    Attributes
    ----------
    frames : list[int]
        Frame indices pooled into this batch.
    angle : float
        Contact angle (degrees) implied by the joint 3D tanh-model fit.
    model_params : dict[str, float]
        Fitted parameters of the 3D hyperbolic tangent model; keys are
        ``"rho1"``, ``"rho2"``, ``"R_eq"``, ``"xi_c"``, ``"yi_c"``,
        ``"zi_c"``, ``"zi_0"``, ``"t1"``, ``"t2"``. The droplet
        horizontal centers ``xi_c`` / ``yi_c`` are reported even if
        the underlying fit fixes them to zero by symmetry.
    xi_grid : ndarray
        x-binning grid centers (Å).
    yi_grid : ndarray
        y-binning grid centers (Å).
    zi_grid : ndarray
        Vertical binning grid centers (Å).
    density : ndarray
        ``(len(xi_grid), len(yi_grid), len(zi_grid))`` binned density
        on the 3D grid.
    """

    frames: list[int]
    angle: float
    model_params: dict[str, float]
    xi_grid: np.ndarray
    yi_grid: np.ndarray
    zi_grid: np.ndarray
    density: np.ndarray


class _AngleResultsMixin:
    """Shared per-batch angle aggregation for the results containers.

    Mixed into the three results dataclasses, each of which exposes a
    ``batches`` list whose elements carry an ``angle`` attribute
    (degrees). Holds only behaviour — the ``batches`` /
    ``method_metadata`` fields stay on the concrete dataclasses so each
    keeps its precise per-batch element type.
    """

    batches: list[Any]

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def per_batch_angles(self) -> np.ndarray:
        """Per-batch contact angle (degrees), in batch order."""
        return np.array([b.angle for b in self.batches])

    @property
    def mean_angle(self) -> float:
        """Mean contact angle across batches (degrees)."""
        if not self.batches:
            return float("nan")
        return float(np.mean(self.per_batch_angles))

    @property
    def std_angle(self) -> float:
        """Standard deviation of the per-batch contact angle (degrees)."""
        if not self.batches:
            return float("nan")
        return float(np.std(self.per_batch_angles))


@dataclass
class TrajectoryResults(_AngleResultsMixin):
    """In-memory results of a :class:`TrajectoryAnalyzer.analyze` run.

    Holds one :class:`BatchResult` per batch produced by the
    :class:`TemporalAggregator`. Within a single Results object all
    batches share the same subclass (:class:`SlicingBatchResult` or
    :class:`WholeBatchResult`), determined by the analyzer's
    :class:`SurfaceFitter` kind.

    Attributes
    ----------
    batches : list[BatchResult]
        Per-batch results, in the order produced by the aggregator.
    method_metadata : dict
        Free-form descriptor of the analyzer configuration (kind,
        droplet geometry, batch size, …) for downstream plotting /
        serialization.
    """

    batches: list[BatchResult]
    method_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoupledFit2DResults(_AngleResultsMixin):
    """In-memory results of a :class:`CoupledFit2DAnalyzer.analyze` run.

    Attributes
    ----------
    batches : list[CoupledFit2DBatchResult]
        Per-batch results, in the order produced by the aggregator.
    method_metadata : dict
        Free-form descriptor (droplet geometry, binning params,
        initial parameters, batch size).
    """

    batches: list[CoupledFit2DBatchResult]
    method_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoupledFit3DResults(_AngleResultsMixin):
    """In-memory results of a :class:`CoupledFit3DAnalyzer.analyze` run.

    Attributes
    ----------
    batches : list[CoupledFit3DBatchResult]
        Per-batch results, in the order produced by the aggregator.
    method_metadata : dict
        Free-form descriptor (droplet geometry — always spherical for
        this analyzer, binning params, initial parameters, batch size).
    """

    batches: list[CoupledFit3DBatchResult]
    method_metadata: dict[str, Any] = field(default_factory=dict)
