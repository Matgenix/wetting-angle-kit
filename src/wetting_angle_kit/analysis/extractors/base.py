"""``InterfaceExtractor`` ABC and the four classmethod factories.

The factories use deferred imports of the concrete extractor classes
to avoid a circular dependency with the sibling ``_rays`` / ``_grid``
modules (which inherit from :class:`InterfaceExtractor`).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np

from wetting_angle_kit.analysis.geometry import DropletGeometry

#: What the downstream :class:`SurfaceFitter` will consume.
SurfaceKind = Literal["slicing", "whole"]

#: Sampling strategy used by the extractor.
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


class InterfaceExtractor(ABC):
    """Abstract base for interface point extractors.

    Concrete extractors are constructed through the classmethod
    factories :meth:`rays_gaussian`, :meth:`rays_binning`,
    :meth:`grid_gaussian`, :meth:`grid_binning`. Direct subclassing is
    supported for custom strategies; the factories cover the built-in
    cases.
    """

    #: Sampling strategy this extractor uses. Set by each concrete subclass.
    sampling: ClassVar[SamplingKind]

    @abstractmethod
    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        surface_kind: SurfaceKind,
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
            slicing modes and the ray-fan layout for whole modes.
        surface_kind : {"slicing", "whole"}
            What the downstream :class:`SurfaceFitter` will consume.
            Determines the output shape (per-slice 2D points vs 3D
            shell). The analyzer enforces ``surface_kind == fitter.kind``
            via :meth:`validate_compatibility` at construction.

        Returns
        -------
        InterfaceData
            ``list[ndarray]`` of ``(M_i, 2)`` per-slice points when
            ``surface_kind="slicing"``; a single ``(N, 3)`` shell when
            ``surface_kind="whole"``.
        """

    @abstractmethod
    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        """Raise if this extractor cannot serve ``(surface_kind, geometry)``.

        Called by :class:`TrajectoryAnalyzer.__init__` so misconfigurations
        fail fast at construction instead of at the first batch.
        """

    @classmethod
    def rays_gaussian(
        cls,
        *,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
        n_rays_sphere: int | None = None,
        delta_polar: float = 8.0,
        points_per_angstrom: float = 1.0,
        density_sigma: float = 3.0,
        cutoff_sigma: float = 5.0,
    ) -> "InterfaceExtractor":
        """Ray fan + Gaussian KDE + per-ray tanh interface fit.

        The interface position along each ray is recovered by smoothing
        atom positions with a Gaussian kernel and fitting a hyperbolic
        tangent profile to the resulting 1D density.

        Required ray-fan parameters depend on the
        ``(surface_kind, droplet_geometry)`` the extractor will be
        paired with:

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
        density_sigma : float, default 3.0
            Gaussian kernel width (Å) for the density smoothing.
            Default tuned for full-atomistic water at room temperature.
        cutoff_sigma : float, default 5.0
            Per-atom kernel truncation in units of ``density_sigma``.
        """
        from wetting_angle_kit.analysis.extractors._rays import (
            _RaysGaussianExtractor,
        )

        return _RaysGaussianExtractor(
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
            n_rays_sphere=n_rays_sphere,
            delta_polar=delta_polar,
            points_per_angstrom=points_per_angstrom,
            density_sigma=density_sigma,
            cutoff_sigma=cutoff_sigma,
        )

    @classmethod
    def rays_binning(
        cls,
        *,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
        n_rays_sphere: int | None = None,
        delta_polar: float = 8.0,
        bin_width: float = 1.0,
        points_per_angstrom: float = 1.0,
    ) -> "InterfaceExtractor":
        """Ray fan + histogram density + per-ray tanh interface fit.

        Same ray-fan geometry as :meth:`rays_gaussian` (see that method
        for the parameter compatibility table) but density along each
        ray is estimated via a 1D histogram rather than a Gaussian
        kernel.

        Parameters
        ----------
        delta_azimuthal, delta_cylinder, n_rays_sphere, delta_polar :
            See :meth:`rays_gaussian`.
        bin_width : float, default 1.0
            Diameter (Å) of the 3D top-hat kernel used at each sample
            position along the ray: atoms within ``bin_width / 2`` of
            a sample contribute uniformly to the density, atoms outside
            do not. The natural analogue of :meth:`rays_gaussian`'s
            ``density_sigma``, but with a hard cutoff instead of a
            smooth fall-off.
        points_per_angstrom : float, default 1.0
            Sampling density along each ray (samples per Å).
        """
        from wetting_angle_kit.analysis.extractors._rays import (
            _RaysBinningExtractor,
        )

        return _RaysBinningExtractor(
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
            n_rays_sphere=n_rays_sphere,
            delta_polar=delta_polar,
            bin_width=bin_width,
            points_per_angstrom=points_per_angstrom,
        )

    @classmethod
    def grid_gaussian(
        cls,
        *,
        grid_params: dict[str, Any] | None = None,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
        density_sigma: float = 3.0,
        cutoff_sigma: float = 5.0,
    ) -> "InterfaceExtractor":
        """3D Gaussian-KDE density on a fixed grid + isocontour extraction.

        Same density estimator as :meth:`rays_gaussian`, evaluated at
        grid cell centres rather than along rays.

        Per-slice in slicing mode: spherical droplets iterate over
        azimuthal angles ``γ ∈ [0°, 180°)`` controlled by
        ``delta_azimuthal``; cylindrical droplets iterate over axial
        steps controlled by ``delta_cylinder``. Each slice produces an
        ``(s, z)`` density grid and one iso-contour. Whole mode builds
        a 3D ``(x, y, z)`` grid centred laterally on the droplet COM
        and runs marching cubes.

        Parameters
        ----------
        grid_params : dict, optional
            Grid spec. For slicing, six keys: ``"xi_0"``, ``"xi_f"``,
            ``"bin_width_x"``, ``"zi_0"``, ``"zi_f"``,
            ``"bin_width_z"``. ``xi_0`` should be negative for a
            centred slice that spans both halves of the diameter. For
            whole, add three more: ``"yi_0"``, ``"yi_f"``,
            ``"bin_width_y"`` (xi/yi grids are in the droplet-centred
            lateral frame; zi stays in the lab frame). If ``None``
            (default), the grid is auto-derived per batch from the
            atom bounding box plus a 5 Å buffer, with cell width set
            to ``density_sigma / 2``.
        delta_azimuthal : float, optional
            Azimuthal step (degrees) between slicing planes for
            ``slicing + spherical``. Required for that case; ignored
            otherwise.
        delta_cylinder : float, optional
            Step (Å) along the cylinder axis between slicing planes
            for ``slicing + cylinder``. Required for that case;
            ignored otherwise.
        density_sigma : float, default 3.0
            Gaussian kernel width (Å) for the KDE.
        cutoff_sigma : float, default 5.0
            Per-atom kernel truncation in units of ``density_sigma``.
        """
        from wetting_angle_kit.analysis.extractors._grid import (
            _GridGaussianExtractor,
        )

        return _GridGaussianExtractor(
            grid_params=dict(grid_params) if grid_params is not None else None,
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
            density_sigma=density_sigma,
            cutoff_sigma=cutoff_sigma,
        )

    @classmethod
    def grid_binning(
        cls,
        *,
        grid_params: dict[str, Any] | None = None,
        delta_azimuthal: float | None = None,
        delta_cylinder: float | None = None,
    ) -> "InterfaceExtractor":
        """Histogram density on a fixed grid + isocontour extraction.

        Same per-slice iteration scheme as :meth:`grid_gaussian` in
        slicing mode (``delta_azimuthal`` for spherical,
        ``delta_cylinder`` for cylinder), but the per-cell density is
        a top-hat histogram count divided by the cell volume.

        For slicing mode, the bin attached to each ``(s, z)`` cell is a
        ``ds × dz × bin_width_x`` box: atoms within
        ``±bin_width_x/2`` of the slice plane contribute. The
        perpendicular slab thickness re-uses ``grid_params["bin_width_x"]``
        — refining the in-plane grid also thins the slab, so coarser
        grids reduce per-bin Poisson noise.

        For whole mode, the 3D cells defined by ``grid_params`` are the
        bins directly (``bin_width_x`` is then unused).

        Parameters
        ----------
        grid_params : dict, optional
            Same shape as :meth:`grid_gaussian`'s ``grid_params``.
            If ``None`` (default), the grid is auto-derived per batch
            from the atom bounding box plus a 5 Å buffer, with a flat
            ``2 Å`` cell width. The histogram estimator is
            intrinsically noisy for single-frame slicing-mode
            analyses (the slab cut leaves few atoms per cell); for
            that case pool multiple frames per batch or supply a
            hand-tuned ``grid_params`` rather than relying on the
            default.
        delta_azimuthal, delta_cylinder : float, optional
            See :meth:`grid_gaussian`.
        """
        from wetting_angle_kit.analysis.extractors._grid import (
            _GridBinningExtractor,
        )

        return _GridBinningExtractor(
            grid_params=dict(grid_params) if grid_params is not None else None,
            delta_azimuthal=delta_azimuthal,
            delta_cylinder=delta_cylinder,
        )
