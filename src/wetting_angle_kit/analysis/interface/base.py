"""Interface-extraction composer and the space-sampling strategy.

This module owns three closely coupled pieces of the interface-finding
subsystem:

- the type aliases :data:`SurfaceKind`, :data:`SamplingKind`, and
  :data:`InterfaceData` that flow through the pipeline;
- :class:`SpaceSampling` — the strategy that decides *where* in 3D
  space density is evaluated (exposed via the factories
  :meth:`SpaceSampling.rays` and :meth:`SpaceSampling.grid`);
- :class:`InterfaceExtractor` — the thin composition layer that pairs
  a :class:`SpaceSampling` with a :class:`DensityEstimator` and
  produces the interface points consumed by
  :class:`SurfaceFitter`.

The concrete sampling implementations (:class:`_RaysSampling`,
:class:`_GridSampling`) live in sibling private modules and are
constructed via the factories on :class:`SpaceSampling`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np

from wetting_angle_kit.analysis.density_estimator import DensityEstimator
from wetting_angle_kit.analysis.geometry import DropletGeometry

# ---------------------------------------------------------------------------
# Type aliases.
# ---------------------------------------------------------------------------

#: What the downstream :class:`SurfaceFitter` will consume.
SurfaceKind = Literal["slicing", "whole"]

#: Tag identifying which sampling strategy a :class:`SpaceSampling`
#: instance implements. Used for tqdm labels and result metadata.
SamplingKind = Literal["rays", "grid"]

#: Interface point set produced by an :class:`InterfaceExtractor` and
#: consumed by :class:`SurfaceFitter` (and, via :class:`WallContext`,
#: by :class:`WallDetector`).
#:
#: - In slicing mode, a list of ``(N_i, 2)`` arrays in the per-slice
#:   ``(x, z)`` plane.
#: - In whole mode, a single ``(N, 3)`` array in the internal
#:   ``(x, y, z)`` frame.
InterfaceData: TypeAlias = list[np.ndarray] | np.ndarray


# ---------------------------------------------------------------------------
# SpaceSampling — strategy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceSampling(ABC):
    """Strategy interface for space-sampling layouts.

    Concrete instances come from one of the classmethod factories
    :meth:`rays` or :meth:`grid`; the abstract :meth:`extract` and
    :meth:`validate_compatibility` methods are dispatched by the
    composing :class:`InterfaceExtractor` after pooling atom positions.
    """

    # kind tag (used in tqdm labels). Set by each
    # concrete subclass.
    kind: ClassVar[SamplingKind]

    @abstractmethod
    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        """Raise if this sampling cannot serve ``(surface_kind, geometry)``.

        Called by :class:`TrajectoryAnalyzer.__init__` so misconfigurations
        fail fast at construction instead of at the first batch.
        """

    @abstractmethod
    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        surface_kind: SurfaceKind,
        density: DensityEstimator,
    ) -> InterfaceData:
        """Build the interface point set for one batch.

        Parameters
        ----------
        liquid_coordinates : ndarray, shape (N, 3)
            Pooled liquid-atom coordinates in the internal frame.
        center_geom : ndarray, shape (3,)
            Geometric droplet center.
        droplet_geometry : DropletGeometry
            Droplet symmetry; drives the per-slice axis choice for
            slicing modes and the ray-fan / grid layout for whole
            modes.
        surface_kind : {"slicing", "whole"}
            What the downstream :class:`SurfaceFitter` will consume.
        density : DensityEstimator
            Density-estimation strategy. The sampling delegates per-cell
            or per-ray-sample density to this strategy.

        Returns
        -------
        InterfaceData
            ``list[ndarray]`` of ``(M_i, 2)`` per-slice points when
            ``surface_kind="slicing"``; a single ``(N, 3)`` shell when
            ``surface_kind="whole"``.
        """

    # ------------------------------------------------------------------
    # Factories.
    # ------------------------------------------------------------------

    @classmethod
    def rays(
        cls,
        *,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
        n_rays_sphere: int | None = None,
        delta_polar: float = 8.0,
        points_per_angstrom: float = 1.0,
    ) -> SpaceSampling:
        """Ray-fan sampling layout.

        Required ray-fan parameters depend on the
        ``(surface_kind, droplet_geometry)`` the sampling is paired
        with:

        ==========================  =========================================
        surface_kind, geometry      required ray params
        ==========================  =========================================
        slicing, spherical          ``delta_azimuthal`` (+ ``delta_polar``)
        slicing, cylinder_x/y       ``delta_cylinder`` (+ ``delta_polar``)
        whole, spherical            ``n_rays_sphere``
        whole, cylinder_x/y         ``delta_cylinder`` (+ ``delta_polar``)
        ==========================  =========================================

        Parameters
        ----------
        delta_azimuthal : float, optional
            Azimuthal step (degrees) between slicing planes for the
            spherical slicing mode.
        delta_cylinder : float, optional
            Step (Å) along the cylinder axis between slices for the
            cylinder modes (both slicing and whole).
        n_rays_sphere : int, optional
            Total number of rays covering the **full sphere** for the
            spherical whole-fit mode. Rays are placed via an equal-area
            Fibonacci ``(cos θ, φ)`` construction so the angular density
            is uniform from south to north pole. Full-sphere (rather
            than upper-hemisphere) coverage is intentional: downward
            rays from the droplet COM traverse the liquid and hit the
            wall plane, producing interface points at the wall — that
            keeps :meth:`WallDetector.min_plus_offset` consistent with
            the physical wall position.
        delta_polar : float, default 8.0
            In-plane ray step (degrees) for every mode that emits rays
            in the ``(x, z)`` plane (i.e. everything except
            whole + spherical).
        points_per_angstrom : float, default 1.0
            Sampling density along each ray (samples per Å).
        """
        from wetting_angle_kit.analysis.interface._rays import _RaysSampling

        return _RaysSampling(
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
            n_rays_sphere=n_rays_sphere,
            delta_polar=delta_polar,
            points_per_angstrom=points_per_angstrom,
        )

    @classmethod
    def grid(
        cls,
        *,
        grid_params: dict[str, Any] | None = None,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
    ) -> SpaceSampling:
        """Fixed-cell grid sampling layout.

        Per-slice in slicing mode: spherical droplets iterate over
        azimuthal angles ``γ ∈ [0°, 180°)`` controlled by
        ``delta_azimuthal``; cylindrical droplets iterate over axial
        steps controlled by ``delta_cylinder``. Each slice produces
        an ``(s, z)`` density grid and one iso-contour. Whole mode
        builds a 3D ``(x, y, z)`` grid centred laterally on the
        droplet COM and runs marching cubes.

        Parameters
        ----------
        grid_params : dict, optional
            Grid spec. For slicing, six keys: ``"xi_0"``, ``"xi_f"``,
            ``"dx"``, ``"zi_0"``, ``"zi_f"``,
            ``"dz"``. ``xi_0`` should be negative for a
            centred slice that spans both halves of the diameter. For
            whole, add three more: ``"yi_0"``, ``"yi_f"``,
            ``"dy"`` (xi/yi grids are in the droplet-centred
            lateral frame; zi stays in the lab frame). If ``None``
            (default), the grid is auto-derived per batch from the
            atom bounding box plus a 5 Å buffer, with cell width set
            to ``density_sigma / 2`` for Gaussian or ``2 Å`` for
            binning.
        delta_azimuthal : float, optional
            Azimuthal step (degrees) between slicing planes for
            ``slicing + spherical``. Required for that case; ignored
            otherwise.
        delta_cylinder : float, optional
            Step (Å) along the cylinder axis between slicing planes
            for ``slicing + cylinder``. Required for that case;
            ignored otherwise.
        """
        from wetting_angle_kit.analysis.interface._grid import _GridSampling

        return _GridSampling(
            grid_params=dict(grid_params) if grid_params is not None else None,
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
        )


# ---------------------------------------------------------------------------
# InterfaceExtractor — composer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class InterfaceExtractor:
    """Composes a sampling layout with a density estimator.

    Parameters
    ----------
    sampling : SpaceSampling
        Space-sampling strategy. Built via
        :meth:`SpaceSampling.rays` or :meth:`SpaceSampling.grid`.
    density : DensityEstimator
        Density-estimation strategy. Built via
        :meth:`DensityEstimator.gaussian` or
        :meth:`DensityEstimator.binning`.

    Examples
    --------
    >>> from wetting_angle_kit.analysis import (
    ...     DensityEstimator, InterfaceExtractor, SpaceSampling,
    ... )
    >>> extractor = InterfaceExtractor(
    ...     sampling=SpaceSampling.rays(
    ...         delta_azimuthal=20.0, delta_polar=8.0,
    ...     ),
    ...     density=DensityEstimator.gaussian(density_sigma=3.0),
    ... )
    """

    sampling: SpaceSampling
    density: DensityEstimator

    @property
    def sampling_kind(self) -> SamplingKind:
        """Tag identifying the sampling layout (``"rays"`` or ``"grid"``)."""
        return self.sampling.kind

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        """Raise if this extractor cannot serve ``(surface_kind, geometry)``.

        Forwards to :meth:`SpaceSampling.validate_compatibility`; the
        sampling owns the validation rules (e.g. ``delta_azimuthal`` is
        required for slicing-spherical rays).
        """
        self.sampling.validate_compatibility(surface_kind, droplet_geometry)

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        surface_kind: SurfaceKind,
    ) -> InterfaceData:
        """Build the interface point set for one batch.

        Delegates to :meth:`SpaceSampling.extract`, threading
        ``self.density`` through.
        """
        return self.sampling.extract(
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            surface_kind=surface_kind,
            density=self.density,
        )
