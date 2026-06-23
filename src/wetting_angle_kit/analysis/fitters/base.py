"""``SurfaceFitter`` ABC + ``FitOutput`` types + factory classmethods.

The factories use deferred imports of the concrete fitter classes to
avoid a circular dependency with the sibling ``_slicing`` / ``_whole``
modules (which inherit from :class:`SurfaceFitter`).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface.base import InterfaceData
from wetting_angle_kit.analysis.results import (
    BatchResult,
    SlicingBatchResult,
    WholeBatchResult,
)

#: Surface-representation kind the fitter consumes. Mirrors
#: :data:`wetting_angle_kit.analysis.interface.SurfaceKind` — the two
#: are kept in sync by the analyzer's compatibility check, which raises
#: if ``extractor.kind != fitter.kind``.
SurfaceKind = Literal["slicing", "whole"]


class FitOutput(ABC):
    """Frames-less per-batch fit output returned by :meth:`SurfaceFitter.fit`.

    The fitter computes the geometric fit and returns one of
    :class:`SlicingFitOutput` or :class:`WholeFitOutput`. The analyzer
    then calls :meth:`to_batch_result` with the batch's frame indices
    to produce the user-facing :class:`BatchResult` — keeping
    bookkeeping (frames) and computation (the fit) separate.
    """

    @abstractmethod
    def to_batch_result(self, frames: list[int]) -> BatchResult:
        """Attach ``frames`` to this fit output and return a BatchResult."""


@dataclass(frozen=True, eq=False, kw_only=True)
class SlicingFitOutput(FitOutput):
    """Output of :meth:`SurfaceFitter.slicing` for one batch.

    Carries the same payload as :class:`SlicingBatchResult` minus
    ``frames``. Field semantics are identical; see that class for
    documentation.
    """

    angle: float
    z_wall: float
    rms_residual: float
    angle_std: float
    per_slice_angles: np.ndarray
    slice_surfaces: list[np.ndarray]
    slice_popts: np.ndarray
    n_slices_total: int
    n_slices_used: int

    def to_batch_result(self, frames: list[int]) -> SlicingBatchResult:
        return SlicingBatchResult(
            frames=frames,
            angle=self.angle,
            z_wall=self.z_wall,
            rms_residual=self.rms_residual,
            angle_std=self.angle_std,
            per_slice_angles=self.per_slice_angles,
            slice_surfaces=self.slice_surfaces,
            slice_popts=self.slice_popts,
            n_slices_total=self.n_slices_total,
            n_slices_used=self.n_slices_used,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class WholeFitOutput(FitOutput):
    """Output of :meth:`SurfaceFitter.whole` for one batch.

    Carries the same payload as :class:`WholeBatchResult` minus
    ``frames``. Field semantics are identical; see that class for
    documentation.
    """

    angle: float
    z_wall: float
    rms_residual: float
    angle_std: float | None
    interface_shell: np.ndarray
    popt: np.ndarray

    def to_batch_result(self, frames: list[int]) -> WholeBatchResult:
        return WholeBatchResult(
            frames=frames,
            angle=self.angle,
            z_wall=self.z_wall,
            rms_residual=self.rms_residual,
            angle_std=self.angle_std,
            interface_shell=self.interface_shell,
            popt=self.popt,
        )


class SurfaceFitter(ABC):
    """Abstract base for contact-angle surface fitters.

    Concrete fitters are constructed through the classmethod factories
    :meth:`slicing` and :meth:`whole`. Direct subclassing is supported
    for custom strategies but the factories cover all built-in cases.
    """

    #: Surface-representation kind this fitter consumes. Set by each
    #: concrete subclass; the analyzer matches this against the chosen
    #: :class:`InterfaceExtractor` at construction time.
    kind: ClassVar[SurfaceKind]

    @abstractmethod
    def fit(
        self,
        interface_data: InterfaceData,
        z_wall: float,
        droplet_geometry: DropletGeometry,
    ) -> FitOutput:
        """Fit the contact angle for one batch.

        Parameters
        ----------
        interface_data : InterfaceData
            Interface point set produced by the
            :class:`InterfaceExtractor`. Per-slice 2D points for
            ``kind="slicing"``; a 3D shell for ``kind="whole"``.
        z_wall : float
            Wall-plane z-coordinate from the :class:`WallDetector`.
        droplet_geometry : DropletGeometry
            Droplet symmetry; controls the geometric model
            (circle per slice / sphere / cylinder).

        Returns
        -------
        FitOutput
            :class:`SlicingFitOutput` for slicing fitters,
            :class:`WholeFitOutput` for whole fitters. The analyzer
            attaches the batch's frame indices via
            :meth:`FitOutput.to_batch_result` to produce the
            user-facing :class:`BatchResult`.
        """

    @abstractmethod
    def validate_compatibility(self, droplet_geometry: DropletGeometry) -> None:
        """Raise if this fitter cannot handle ``droplet_geometry``.

        Called by :class:`TrajectoryAnalyzer.__init__`. The kind
        compatibility (slicing vs whole) is enforced separately at
        the analyzer level by matching :attr:`SurfaceFitter.kind`
        against the extractor's chosen ``surface_kind``.
        """

    @classmethod
    def slicing(
        cls,
        *,
        surface_filter_offset: float = 2.0,
    ) -> "SurfaceFitter":
        """Per-slice algebraic circle fits, averaged across slices.

        Each slice's 2D interface points are filtered to
        ``z > z_wall + surface_filter_offset`` (to exclude
        wall-adjacent density distortions), an algebraic Taubin circle
        is fit to the kept points, and the contact angle is the
        angle of intersection between that circle and the line
        ``z = z_wall``. The batch angle is the mean over slices;
        :attr:`BatchResult.angle_std` is the empirical std across
        slices.

        Parameters
        ----------
        surface_filter_offset : float, default 2.0
            Vertical offset above ``z_wall`` (Å) below which interface
            points are excluded from the circle fit. This is distinct
            from any offset baked into the :class:`WallDetector`: this
            offset is a fit-quality knob for the per-slice circle, and
            the wall detector's offset (if it uses one, e.g.
            :meth:`WallDetector.min_plus_offset`) defines where the
            wall plane sits.
        """
        from wetting_angle_kit.analysis.fitters._slicing import _SlicingFitter

        return _SlicingFitter(surface_filter_offset=surface_filter_offset)

    @classmethod
    def whole(
        cls,
        *,
        surface_filter_offset: float = 2.0,
        bootstrap_samples: int = 0,
    ) -> "SurfaceFitter":
        """Algebraic sphere or cylinder fit to the 3D interface shell.

        Spherical droplets get a sphere fit; cylindrical droplets get
        a circular-cylinder fit whose axis is parallel to ``y``
        (internal frame, post axis-swap for ``cylinder_x``). The
        contact angle follows from the cap geometry:
        ``cos θ = (z_wall - z_center) / R``.

        Parameters
        ----------
        surface_filter_offset : float, default 2.0
            Vertical offset above ``z_wall`` (Å) below which shell
            points are excluded from the geometric fit. Same role as
            in :meth:`slicing`: distinct from the wall detector's
            offset.
        bootstrap_samples : int, default 0
            If positive, the fit is repeated on this many bootstrap
            resamples of the filtered shell, and the resulting std
            of the angles is reported as
            :attr:`BatchResult.angle_std`. ``0`` disables bootstrap;
            the field is then ``None`` in the returned
            :class:`WholeBatchResult`.
        """
        from wetting_angle_kit.analysis.fitters._whole import _WholeFitter

        return _WholeFitter(
            surface_filter_offset=surface_filter_offset,
            bootstrap_samples=bootstrap_samples,
        )
