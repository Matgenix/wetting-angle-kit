"""Interface extractors: build the interface point set from raw atoms.

An :class:`InterfaceExtractor` converts pooled liquid-atom coordinates
into one of two output shapes, determined by the :class:`SurfaceFitter`
the analyzer is paired with:

- ``surface_kind="slicing"`` → a list of per-slice ``(M_i, 2)`` arrays
  in the slice ``(x, z)`` plane;
- ``surface_kind="whole"`` → a single ``(N, 3)`` shell array in the
  internal ``(x, y, z)`` frame.

Extractors are constructed through classmethod factories on the base
class — each factory configures one sampling + density-kernel
combination::

    InterfaceExtractor.rays_gaussian(...)   # ray fan + Gaussian KDE + tanh
    InterfaceExtractor.rays_binning(...)    # ray fan + histogram bins + tanh
    InterfaceExtractor.grid_gaussian(...)   # 2D KDE map + isocontour (slicing only)
    InterfaceExtractor.grid_binning(...)    # 2D histogram map + isocontour
                                            # (slicing only)

The pairing between the chosen extractor and the analyzer's
:class:`SurfaceFitter` is validated at :class:`TrajectoryAnalyzer`
construction via :meth:`InterfaceExtractor.validate_compatibility`.

Algorithm bodies are stubbed (``raise NotImplementedError``) at this
skeleton stage; only constructor surfaces and compatibility checks are
implemented here. Density / tanh-fit primitives will be pulled into
``analysis/_density.py`` when porting begins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from wetting_angle_kit.analysis._density import (
    MIN_POINTS_PER_RAY,
    DensityFieldProtocol,
    GaussianDensityField,
    HistogramDensityField,
    fit_tanh_profiles_batched,
)
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
            Total number of rays in the upper hemisphere for the
            spherical whole-fit mode. Rays are placed via an equal-area
            ``(cos θ, φ)`` construction to avoid pole bias.
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
        return _GridBinningExtractor(grid_params=dict(grid_params))


_GRID_KEYS_2D = frozenset({"xi_0", "xi_f", "nbins_xi", "zi_0", "zi_f", "nbins_zi"})
_GRID_KEYS_3D = _GRID_KEYS_2D | {"yi_0", "yi_f", "nbins_yi"}


def _validate_grid_params(
    *,
    name: str,
    grid_params: dict[str, Any],
    surface_kind: SurfaceKind,
) -> None:
    """Shared validation for the two grid-based extractors.

    Checks that ``grid_params`` has the right dimensionality for
    ``surface_kind`` and, for whole-kind, that ``scikit-image`` is
    importable (it is the marching-cubes backend).
    """
    if surface_kind == "slicing":
        missing = _GRID_KEYS_2D - grid_params.keys()
        if missing:
            raise ValueError(
                f"{name} for slicing requires a 2D grid_params; missing "
                f"keys: {sorted(missing)}."
            )
        return
    # surface_kind == "whole"
    try:
        import skimage.measure  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{name} for whole-kind extraction requires scikit-image "
            "(used for marching_cubes). Install with: "
            "pip install 'wetting-angle-kit[grid3d]'."
        ) from e
    missing = _GRID_KEYS_3D - grid_params.keys()
    if missing:
        raise ValueError(
            f"{name} for whole requires a 3D grid_params; missing keys: "
            f"{sorted(missing)}."
        )


def _fibonacci_hemisphere_directions(n: int) -> np.ndarray:
    """Equal-area Fibonacci-spiral directions on the upper hemisphere.

    ``cos θ`` is uniformly spaced over ``[0, 1]`` (so the surface
    density is uniform on the sphere) and ``φ`` is incremented by the
    golden angle for low-discrepancy azimuthal coverage. ``i = 0``
    sits at the horizon (``cos θ = 0``) and ``i = n - 1`` at the pole
    (``cos θ = 1``).

    Parameters
    ----------
    n : int
        Number of directions.

    Returns
    -------
    ndarray, shape (n, 3)
        Unit direction vectors.
    """
    if n <= 0:
        return np.empty((0, 3))
    i = np.arange(n, dtype=np.float64)
    cos_theta = i / (n - 1) if n > 1 else np.array([1.0])
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = (i * golden_angle) % (2.0 * np.pi)
    return np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )


def _validate_rays_params(
    *,
    name: str,
    delta_azimuthal: float | None,
    delta_cylinder: float | None,
    n_rays_sphere: int | None,
    surface_kind: SurfaceKind,
    droplet_geometry: DropletGeometry,
) -> None:
    """Shared validation for the two ray-based extractors.

    Both ``rays_gaussian`` and ``rays_binning`` use the same ray-fan
    parameter set; only the density estimator differs.
    """
    if surface_kind == "slicing":
        if droplet_geometry.is_spherical and delta_azimuthal is None:
            raise ValueError(f"{name} for slicing+spherical requires delta_azimuthal.")
        if droplet_geometry.is_cylinder and delta_cylinder is None:
            raise ValueError(
                f"{name} for slicing+{droplet_geometry.name} requires delta_cylinder."
            )
    elif surface_kind == "whole":
        if droplet_geometry.is_spherical and n_rays_sphere is None:
            raise ValueError(f"{name} for whole+spherical requires n_rays_sphere.")
        if droplet_geometry.is_cylinder and delta_cylinder is None:
            raise ValueError(
                f"{name} for whole+{droplet_geometry.name} requires delta_cylinder."
            )


def _ray_slice_in_plane(
    field: DensityFieldProtocol,
    center: np.ndarray,
    gamma: float,
    max_dist: float,
    distances: np.ndarray,
    delta_polar: float,
) -> np.ndarray:
    """Per-slice ``(R, 2)`` interface from a tilted ray fan.

    Matches the legacy ``SurfaceDefinition.analyze_lines`` body but
    parameterised on a generic :class:`DensityFieldProtocol` so both
    ``rays_gaussian`` and ``rays_binning`` can share the geometry.
    """
    beta = np.linspace(0, 360, int(360 / delta_polar), endpoint=False)
    cos_beta = np.cos(np.deg2rad(beta))
    sin_beta = np.sin(np.deg2rad(beta))
    cos_gamma = np.cos(np.deg2rad(gamma))
    sin_gamma = np.sin(np.deg2rad(gamma))
    directions = np.column_stack((cos_beta * cos_gamma, cos_beta * sin_gamma, sin_beta))
    positions_rm = (
        center[None, None, :] + distances[None, :, None] * directions[:, None, :]
    )
    density_flat = field.evaluate(positions_rm.reshape(-1, 3))
    densities = density_flat.reshape(len(beta), len(distances))
    interface_re = fit_tanh_profiles_batched(distances, densities, max_dist=max_dist)
    x_proj = cos_beta * interface_re + center[0]
    z_proj = sin_beta * interface_re + center[2]
    return np.column_stack([x_proj, z_proj])


def _extract_rays(
    *,
    field: DensityFieldProtocol,
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
    max_dist: float,
    surface_kind: SurfaceKind,
    points_per_angstrom: float,
    delta_azimuthal: float | None,
    delta_cylinder: float | None,
    n_rays_sphere: int | None,
    delta_polar: float,
) -> InterfaceData:
    """Dispatch a ray-fan extraction over the four ``(kind, geometry)`` cells.

    Shared by :class:`_RaysGaussianExtractor` and
    :class:`_RaysBinningExtractor` — only the density evaluator
    differs between them, so the geometry, sampling cadence, and
    tanh-fit invocation all live here.
    """
    n_samples = max(int(max_dist * points_per_angstrom), MIN_POINTS_PER_RAY)
    distances = np.linspace(0.0, max_dist, n_samples)

    if surface_kind == "slicing":
        if droplet_geometry.is_spherical:
            assert delta_azimuthal is not None
            n_slices = int(180 / delta_azimuthal)
            gammas = np.linspace(0.0, 180.0, n_slices)
            return [
                _ray_slice_in_plane(
                    field, center_geom, float(g), max_dist, distances, delta_polar
                )
                for g in gammas
            ]
        # cylinder_*: y-step slice fan
        assert delta_cylinder is not None
        y_vals = liquid_coordinates[:, 1]
        ys = np.arange(float(y_vals.min()), float(y_vals.max()), delta_cylinder)
        slices: list[np.ndarray] = []
        for y in ys:
            slice_center = np.array([center_geom[0], float(y), center_geom[2]])
            slices.append(
                _ray_slice_in_plane(
                    field, slice_center, 0.0, max_dist, distances, delta_polar
                )
            )
        return slices

    # surface_kind == "whole"
    if droplet_geometry.is_spherical:
        assert n_rays_sphere is not None
        directions = _fibonacci_hemisphere_directions(n_rays_sphere)
        positions_rm = (
            center_geom[None, None, :]
            + distances[None, :, None] * directions[:, None, :]
        )
        density_flat = field.evaluate(positions_rm.reshape(-1, 3))
        densities = density_flat.reshape(len(directions), len(distances))
        interface_re = fit_tanh_profiles_batched(
            distances, densities, max_dist=max_dist
        )
        return center_geom[None, :] + interface_re[:, None] * directions

    # whole + cylinder_*: pool a per-y ray fan into a 3D shell.
    assert delta_cylinder is not None
    y_vals = liquid_coordinates[:, 1]
    ys = np.arange(float(y_vals.min()), float(y_vals.max()), delta_cylinder)
    beta = np.linspace(0, 360, int(360 / delta_polar), endpoint=False)
    cos_beta = np.cos(np.deg2rad(beta))
    sin_beta = np.sin(np.deg2rad(beta))
    cyl_directions = np.column_stack([cos_beta, np.zeros_like(beta), sin_beta])
    shells: list[np.ndarray] = []
    for y in ys:
        slice_center = np.array([center_geom[0], float(y), center_geom[2]])
        positions_rm = (
            slice_center[None, None, :]
            + distances[None, :, None] * cyl_directions[:, None, :]
        )
        density_flat = field.evaluate(positions_rm.reshape(-1, 3))
        densities = density_flat.reshape(len(beta), len(distances))
        interface_re = fit_tanh_profiles_batched(
            distances, densities, max_dist=max_dist
        )
        points = np.column_stack(
            [
                cos_beta * interface_re + slice_center[0],
                np.full(len(beta), float(y)),
                sin_beta * interface_re + slice_center[2],
            ]
        )
        shells.append(points)
    return np.concatenate(shells, axis=0) if shells else np.empty((0, 3))


@dataclass(frozen=True, eq=False, kw_only=True)
class _RaysGaussianExtractor(InterfaceExtractor):
    """Concrete extractor for :meth:`InterfaceExtractor.rays_gaussian`."""

    sampling: ClassVar[SamplingKind] = "rays"

    delta_azimuthal: float | None
    delta_cylinder: float | None
    n_rays_sphere: int | None
    delta_polar: float
    points_per_angstrom: float
    density_sigma: float
    cutoff_sigma: float

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        _validate_rays_params(
            name="rays_gaussian",
            delta_azimuthal=self.delta_azimuthal,
            delta_cylinder=self.delta_cylinder,
            n_rays_sphere=self.n_rays_sphere,
            surface_kind=surface_kind,
            droplet_geometry=droplet_geometry,
        )

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        max_dist: float,
        surface_kind: SurfaceKind,
    ) -> InterfaceData:
        field = GaussianDensityField(
            atom_coords=liquid_coordinates,
            density_sigma=self.density_sigma,
            cutoff_sigma=self.cutoff_sigma,
        )
        return _extract_rays(
            field=field,
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            max_dist=max_dist,
            surface_kind=surface_kind,
            points_per_angstrom=self.points_per_angstrom,
            delta_azimuthal=self.delta_azimuthal,
            delta_cylinder=self.delta_cylinder,
            n_rays_sphere=self.n_rays_sphere,
            delta_polar=self.delta_polar,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class _RaysBinningExtractor(InterfaceExtractor):
    """Concrete extractor for :meth:`InterfaceExtractor.rays_binning`."""

    sampling: ClassVar[SamplingKind] = "rays"

    delta_azimuthal: float | None
    delta_cylinder: float | None
    n_rays_sphere: int | None
    delta_polar: float
    bin_width: float
    points_per_angstrom: float

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        _validate_rays_params(
            name="rays_binning",
            delta_azimuthal=self.delta_azimuthal,
            delta_cylinder=self.delta_cylinder,
            n_rays_sphere=self.n_rays_sphere,
            surface_kind=surface_kind,
            droplet_geometry=droplet_geometry,
        )

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        max_dist: float,
        surface_kind: SurfaceKind,
    ) -> InterfaceData:
        field = HistogramDensityField(
            atom_coords=liquid_coordinates,
            bin_width=self.bin_width,
        )
        return _extract_rays(
            field=field,
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            max_dist=max_dist,
            surface_kind=surface_kind,
            points_per_angstrom=self.points_per_angstrom,
            delta_azimuthal=self.delta_azimuthal,
            delta_cylinder=self.delta_cylinder,
            n_rays_sphere=self.n_rays_sphere,
            delta_polar=self.delta_polar,
        )


def _project_atoms_to_rz(
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse 3D atom coordinates to 2D ``(r, z)`` via droplet symmetry.

    For spherical droplets, ``r = sqrt((x - cx)² + (y - cy)²)``.
    For cylinder droplets (axis along ``y`` in the internal frame after
    the ``cylinder_x`` axis swap), ``r = |x - cx|``. ``z`` is kept in
    the lab frame so the wall position retains physical meaning.
    """
    dx = liquid_coordinates[:, 0] - center_geom[0]
    if droplet_geometry.is_spherical:
        dy = liquid_coordinates[:, 1] - center_geom[1]
        r = np.hypot(dx, dy)
    else:
        r = np.abs(dx)
    return r, liquid_coordinates[:, 2]


def _build_2d_density_grid(
    atoms_r: np.ndarray,
    atoms_z: np.ndarray,
    grid_params: dict[str, Any],
    droplet_geometry: DropletGeometry,
    *,
    smooth_sigma: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D density grid in ``(r, z)``.

    Volume normalisation:

    - **Spherical:** ``dV = 2π r dxi dzi`` per cell (annular shell).
      Required so the recovered isocontour isn't pulled toward
      smaller ``r``.
    - **Cylinder:** ``dV = dxi dzi`` per cell (constant axial extent).
      The cylinder-axis length cancels out for an isocontour at a
      fraction of the max, so the simpler normalisation is used.

    Returns ``(r_centers, z_centers, density)`` with
    ``density.shape == (n_r_cells, n_z_cells)``.
    """
    r_edges = np.linspace(
        float(grid_params["xi_0"]),
        float(grid_params["xi_f"]),
        int(grid_params["nbins_xi"]),
    )
    z_edges = np.linspace(
        float(grid_params["zi_0"]),
        float(grid_params["zi_f"]),
        int(grid_params["nbins_zi"]),
    )
    counts, _, _ = np.histogram2d(atoms_r, atoms_z, bins=(r_edges, z_edges))
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dr = float(r_edges[1] - r_edges[0])
    dz = float(z_edges[1] - z_edges[0])

    if droplet_geometry.is_spherical:
        # Avoid division by zero at the innermost row (r=0); the contour
        # at fraction-of-max never traverses that point anyway.
        dV_per_row = 2.0 * np.pi * np.maximum(r_centers, 0.5 * dr) * dr * dz
        density = counts / dV_per_row[:, None]
    else:
        density = counts / (dr * dz)

    if smooth_sigma is not None and smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        sigma_cells = (smooth_sigma / dr, smooth_sigma / dz)
        density = gaussian_filter(density, sigma=sigma_cells)
    return r_centers, z_centers, density


def _extract_isocontour_2d(
    r_centers: np.ndarray,
    z_centers: np.ndarray,
    density: np.ndarray,
    *,
    fraction_of_bulk: float = 0.5,
    bulk_percentile: float = 95.0,
) -> np.ndarray:
    """Return the longest density iso-line as ``(M, 2)`` ``(r, z)`` points.

    The contour level is ``fraction_of_bulk * percentile(density,
    bulk_percentile)`` — using a high percentile rather than ``max``
    makes the bulk estimate robust to the Poisson spikes that the
    ``dV_per_row ∝ 1/r`` normalisation can introduce in small-``r``
    bins.
    """
    from skimage.measure import find_contours

    if density.size == 0 or float(density.max()) <= 0:
        return np.empty((0, 2))
    bulk = float(np.percentile(density, bulk_percentile))
    if bulk <= 0:
        return np.empty((0, 2))
    level = fraction_of_bulk * bulk
    # ``no-untyped-call`` fires only when scikit-image is not installed
    # in the type-check env; ``unused-ignore`` keeps the comment tolerant
    # when it IS installed and the call resolves to a typed function.
    contours = find_contours(density, level)  # type: ignore[no-untyped-call,unused-ignore]
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)
    # find_contours returns (row, col) fractional pixel indices, where
    # row indexes axis 0 (= r) and col indexes axis 1 (= z).
    dr = (float(r_centers[-1]) - float(r_centers[0])) / max(len(r_centers) - 1, 1)
    dz = (float(z_centers[-1]) - float(z_centers[0])) / max(len(z_centers) - 1, 1)
    r_phys = float(r_centers[0]) + dr * longest[:, 0]
    z_phys = float(z_centers[0]) + dz * longest[:, 1]
    return np.column_stack([r_phys, z_phys])


def _extract_grid_slicing(
    *,
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
    grid_params: dict[str, Any],
    smooth_sigma: float | None,
) -> list[np.ndarray]:
    """Build a ``(r, z)`` density map + isocontour for a slicing-mode grid extractor.

    Returns a single-element list since the symmetry-collapsed
    ``(r, z)`` reduction makes the slice axis disappear; the downstream
    :class:`SlicingFitter` runs one Kasa circle fit on that contour.
    """
    r, z = _project_atoms_to_rz(liquid_coordinates, center_geom, droplet_geometry)
    r_centers, z_centers, density = _build_2d_density_grid(
        r, z, grid_params, droplet_geometry, smooth_sigma=smooth_sigma
    )
    contour = _extract_isocontour_2d(r_centers, z_centers, density)
    return [contour]


# eq=False avoids the auto __eq__ tripping on the dict field; equality
# between extractor instances is not a use case the package needs.
@dataclass(frozen=True, eq=False, kw_only=True)
class _GridGaussianExtractor(InterfaceExtractor):
    """Concrete extractor for :meth:`InterfaceExtractor.grid_gaussian`."""

    sampling: ClassVar[SamplingKind] = "grid"

    grid_params: dict[str, Any]
    density_sigma: float
    cutoff_sigma: float

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        _validate_grid_params(
            name="grid_gaussian",
            grid_params=self.grid_params,
            surface_kind=surface_kind,
        )

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        max_dist: float,
        surface_kind: SurfaceKind,
    ) -> InterfaceData:
        if surface_kind != "slicing":
            raise NotImplementedError(
                "grid_gaussian whole-kind extraction (3D grid + marching "
                "cubes) lands in Phase 8."
            )
        return _extract_grid_slicing(
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            grid_params=self.grid_params,
            smooth_sigma=self.density_sigma,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class _GridBinningExtractor(InterfaceExtractor):
    """Concrete extractor for :meth:`InterfaceExtractor.grid_binning`."""

    sampling: ClassVar[SamplingKind] = "grid"

    grid_params: dict[str, Any]

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        _validate_grid_params(
            name="grid_binning",
            grid_params=self.grid_params,
            surface_kind=surface_kind,
        )

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        max_dist: float,
        surface_kind: SurfaceKind,
    ) -> InterfaceData:
        if surface_kind != "slicing":
            raise NotImplementedError(
                "grid_binning whole-kind extraction (3D grid + marching "
                "cubes) lands in Phase 8."
            )
        return _extract_grid_slicing(
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            grid_params=self.grid_params,
            smooth_sigma=None,
        )
