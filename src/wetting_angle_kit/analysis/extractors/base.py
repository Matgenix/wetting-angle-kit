"""``InterfaceExtractor`` ABC and the four classmethod factories.

The factories use deferred imports of the concrete extractor classes
to avoid a circular dependency with the sibling ``_rays`` / ``_grid``
modules (which inherit from :class:`InterfaceExtractor`).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import numpy as np

from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.wall import InterfaceData

#: What the downstream :class:`SurfaceFitter` will consume.
SurfaceKind = Literal["slicing", "whole"]

#: Sampling strategy used by the extractor.
SamplingKind = Literal["rays", "grid"]


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
        max_dist: float,
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
        max_dist : float
            Maximum radial distance sampled along each ray (Å).
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
        grid_params: dict[str, Any],
        density_sigma: float = 3.0,
        cutoff_sigma: float = 5.0,
    ) -> "InterfaceExtractor":
        """Gaussian-KDE density grid + isocontour interface extraction.

        Supports both slicing and whole fitters:

        - For ``surface_kind="slicing"``, the grid is 2D in the slice
          ``(x, z)`` plane and a marching-squares-style isocontour gives
          one ``(M, 2)`` interface curve per slice.
        - For ``surface_kind="whole"``, the grid is 3D in
          ``(x, y, z)`` and the interface shell is recovered by
          :func:`skimage.measure.marching_cubes`. This requires the
          optional ``grid3d`` extra (``scikit-image``); construction
          via :class:`TrajectoryAnalyzer` raises a clear
          :class:`ImportError` if it is missing.

        Parameters
        ----------
        grid_params : dict
            Grid spec. For slicing, six keys: ``"xi_0"``, ``"xi_f"``,
            ``"nbins_xi"``, ``"zi_0"``, ``"zi_f"``, ``"nbins_zi"``.
            For whole, add three more: ``"yi_0"``, ``"yi_f"``,
            ``"nbins_yi"``.
        density_sigma : float, default 3.0
            Gaussian kernel width (Å) for the density smoothing.
        cutoff_sigma : float, default 5.0
            Per-atom kernel truncation in units of ``density_sigma``.
        """
        from wetting_angle_kit.analysis.extractors._grid import (
            _GridGaussianExtractor,
        )

        return _GridGaussianExtractor(
            grid_params=dict(grid_params),
            density_sigma=density_sigma,
            cutoff_sigma=cutoff_sigma,
        )

    @classmethod
    def grid_binning(
        cls,
        *,
        grid_params: dict[str, Any],
    ) -> "InterfaceExtractor":
        """Histogram density grid + isocontour interface extraction.

        Same dimensionality + dependency rules as
        :meth:`grid_gaussian`: 2D grid for slicing, 3D grid + marching
        cubes (via optional ``scikit-image``) for whole.

        Parameters
        ----------
        grid_params : dict
            Grid spec; see :meth:`grid_gaussian` for the required keys.
        """
        from wetting_angle_kit.analysis.extractors._grid import (
            _GridBinningExtractor,
        )

        return _GridBinningExtractor(grid_params=dict(grid_params))
