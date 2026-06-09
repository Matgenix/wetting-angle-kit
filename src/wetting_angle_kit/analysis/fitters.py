"""Surface fitters: derive a contact angle from interface points + wall.

A :class:`SurfaceFitter` consumes an interface point set produced by an
:class:`InterfaceExtractor` plus a wall z-coordinate produced by a
:class:`WallDetector`, and returns one :class:`BatchResult` per call
holding the contact angle and fit diagnostics.

Two fitter kinds are supported:

- ``slicing``: one algebraic-circle fit per slice in the slice's
  ``(x, z)`` plane, then a mean across slices.
- ``whole``: one algebraic-sphere fit (spherical droplet) or
  algebraic-cylinder fit (cylindrical droplet) to the 3D interface
  shell.

Users construct fitters through classmethod factories on the base
class::

    SurfaceFitter.slicing()
    SurfaceFitter.whole(bootstrap_samples=0)

Algorithm bodies are stubbed (``raise NotImplementedError``) at this
skeleton stage; only constructor surfaces and compatibility checks are
implemented here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.results import (
    BatchResult,
    SlicingBatchResult,
    WholeBatchResult,
)
from wetting_angle_kit.analysis.wall import InterfaceData

#: Surface-representation kind the fitter consumes. Mirrors
#: :data:`wetting_angle_kit.analysis.extractors.SurfaceKind` — the two
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
        wall-adjacent density distortions), an algebraic Kasa circle
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
        return _WholeFitter(
            surface_filter_offset=surface_filter_offset,
            bootstrap_samples=bootstrap_samples,
        )


def _kasa_circle_fit_2d(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit in 2D.

    Linearises ``(x - xc)^2 + (z - zc)^2 = R^2`` into
    ``2 xc x + 2 zc z + c = x^2 + z^2`` with ``c = R^2 - xc^2 - zc^2``
    and solves with :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    x, z : ndarray
        2D point coordinates.

    Returns
    -------
    (xc, zc, R) : tuple of float
        Fitted circle centre and radius.

    Raises
    ------
    np.linalg.LinAlgError
        If the points are collinear (rank-deficient system).
    ValueError
        If the algebraic solution gives a non-positive ``R^2``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    a_matrix = np.column_stack((2.0 * x, 2.0 * z, np.ones_like(x)))
    rhs = x * x + z * z
    sol, _, _, _ = np.linalg.lstsq(a_matrix, rhs, rcond=None)
    xc, zc, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + xc * xc + zc * zc
    if r_sq <= 0.0:
        raise ValueError(
            f"Algebraic circle fit produced non-positive R^2 ({r_sq:.3g}); "
            "the points are likely degenerate."
        )
    return xc, zc, float(np.sqrt(r_sq))


@dataclass(frozen=True, eq=False, kw_only=True)
class _SlicingFitter(SurfaceFitter):
    """Concrete fitter for :meth:`SurfaceFitter.slicing`."""

    kind: ClassVar[SurfaceKind] = "slicing"

    surface_filter_offset: float

    def validate_compatibility(self, droplet_geometry: DropletGeometry) -> None:
        # Slicing handles all three geometries (spherical and both
        # cylinder orientations); nothing geometry-specific to reject.
        return None

    def fit(
        self,
        interface_data: InterfaceData,
        z_wall: float,
        droplet_geometry: DropletGeometry,
    ) -> SlicingFitOutput:
        if not isinstance(interface_data, list):
            raise TypeError(
                "slicing fitter expects a list of per-slice (M, 2) arrays; "
                f"got {type(interface_data).__name__}."
            )

        z_filter = z_wall + self.surface_filter_offset
        per_slice_angles: list[float] = []
        slice_surfaces: list[np.ndarray] = []
        slice_popts: list[np.ndarray] = []
        slice_rms_residuals: list[float] = []

        for surf in interface_data:
            if surf.size == 0:
                continue
            kept = surf[surf[:, 1] > z_filter]
            # Need at least 3 non-collinear points to fit a circle.
            if len(kept) < 3:
                continue
            try:
                xc, zc, radius = _kasa_circle_fit_2d(kept[:, 0], kept[:, 1])
            except (np.linalg.LinAlgError, ValueError):
                continue
            # Contact angle from circle / wall-line intersection:
            # ``cos θ = (z_wall - z_center) / R``. Drop slices where
            # the fitted circle doesn't intersect the wall.
            delta_z = z_wall - zc
            if abs(delta_z) >= radius:
                continue
            angle = float(np.degrees(np.arccos(delta_z / radius)))
            # Per-slice RMS of the circle-fit residuals (Å). The
            # batch-level rms_residual reported in
            # :class:`SlicingBatchResult` is the mean across slices.
            radii = np.hypot(kept[:, 0] - xc, kept[:, 1] - zc)
            rms = float(np.sqrt(np.mean((radii - radius) ** 2)))

            per_slice_angles.append(angle)
            slice_surfaces.append(surf)
            slice_popts.append(np.array([xc, zc, radius, z_wall]))
            slice_rms_residuals.append(rms)

        if not per_slice_angles:
            raise RuntimeError(
                "slicing fit: no slice produced a valid contact angle "
                "after filtering and circle fitting."
            )

        angles_arr = np.asarray(per_slice_angles, dtype=float)
        return SlicingFitOutput(
            angle=float(np.mean(angles_arr)),
            z_wall=z_wall,
            rms_residual=float(np.mean(slice_rms_residuals)),
            angle_std=float(np.std(angles_arr)),
            per_slice_angles=angles_arr,
            slice_surfaces=slice_surfaces,
            slice_popts=np.asarray(slice_popts, dtype=float),
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class _WholeFitter(SurfaceFitter):
    """Concrete fitter for :meth:`SurfaceFitter.whole`."""

    kind: ClassVar[SurfaceKind] = "whole"

    surface_filter_offset: float
    bootstrap_samples: int

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be >= 0; got {self.bootstrap_samples}."
            )

    def validate_compatibility(self, droplet_geometry: DropletGeometry) -> None:
        # Whole-fit covers spherical (sphere fit) and both cylinder
        # orientations (cylinder fit with the standard axis swap);
        # nothing geometry-specific to reject.
        return None

    def fit(
        self,
        interface_data: InterfaceData,
        z_wall: float,
        droplet_geometry: DropletGeometry,
    ) -> WholeFitOutput:
        raise NotImplementedError("whole surface fit not implemented in skeleton.")
