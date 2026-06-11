"""Grid-based extractor implementations + shared density/isocontour helpers."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis.extractors.base import (
    InterfaceExtractor,
    SamplingKind,
    SurfaceKind,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.wall import InterfaceData

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


def _build_3d_density_grid(
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    grid_params: dict[str, Any],
    *,
    smooth_sigma: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D density grid centered laterally on the droplet COM.

    The ``(x, y)`` coordinates are recentered on ``(center_geom[0],
    center_geom[1])`` so the user can specify symmetric grid bounds
    (e.g. ``xi_0 = -30, xi_f = 30``) regardless of where the droplet
    sits in the box after PBC recentering. ``z`` stays in the lab
    frame so the wall position retains physical meaning.

    Returns ``(x_centers, y_centers, z_centers, density)`` with
    ``density.shape == (n_x_cells, n_y_cells, n_z_cells)``.
    """
    x_edges = np.linspace(
        float(grid_params["xi_0"]),
        float(grid_params["xi_f"]),
        int(grid_params["nbins_xi"]),
    )
    y_edges = np.linspace(
        float(grid_params["yi_0"]),
        float(grid_params["yi_f"]),
        int(grid_params["nbins_yi"]),
    )
    z_edges = np.linspace(
        float(grid_params["zi_0"]),
        float(grid_params["zi_f"]),
        int(grid_params["nbins_zi"]),
    )

    atoms_centered = liquid_coordinates - np.array(
        [center_geom[0], center_geom[1], 0.0]
    )
    counts, _ = np.histogramdd(atoms_centered, bins=(x_edges, y_edges, z_edges))
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    dz = float(z_edges[1] - z_edges[0])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    density = counts / (dx * dy * dz)

    if smooth_sigma is not None and smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        sigma_cells = (
            smooth_sigma / dx,
            smooth_sigma / dy,
            smooth_sigma / dz,
        )
        density = gaussian_filter(density, sigma=sigma_cells)
    return x_centers, y_centers, z_centers, density


def _extract_isosurface_3d(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    density: np.ndarray,
    center_geom: np.ndarray,
    *,
    fraction_of_bulk: float = 0.5,
    bulk_percentile: float = 95.0,
) -> np.ndarray:
    """Run marching cubes and return ``(N, 3)`` shell points in the internal frame.

    The shell is shifted back by ``(center_geom[0], center_geom[1], 0)``
    so its coordinates match the convention used by the rays-whole
    extractor: absolute internal-frame positions, not droplet-centered.
    """
    from skimage.measure import marching_cubes

    if density.size == 0 or float(density.max()) <= 0:
        return np.empty((0, 3))
    bulk = float(np.percentile(density, bulk_percentile))
    if bulk <= 0:
        return np.empty((0, 3))
    level = fraction_of_bulk * bulk
    try:
        # ``no-untyped-call`` fires when scikit-image is not installed
        # in the type-check env; ``unused-ignore`` keeps the comment
        # tolerant when it is.
        verts, _faces, _normals, _values = marching_cubes(  # type: ignore[no-untyped-call,unused-ignore]
            density, level
        )
    except (RuntimeError, ValueError):
        return np.empty((0, 3))

    dx = float(x_centers[-1] - x_centers[0]) / max(len(x_centers) - 1, 1)
    dy = float(y_centers[-1] - y_centers[0]) / max(len(y_centers) - 1, 1)
    dz = float(z_centers[-1] - z_centers[0]) / max(len(z_centers) - 1, 1)
    x_phys = float(x_centers[0]) + dx * verts[:, 0] + float(center_geom[0])
    y_phys = float(y_centers[0]) + dy * verts[:, 1] + float(center_geom[1])
    z_phys = float(z_centers[0]) + dz * verts[:, 2]
    return np.column_stack([x_phys, y_phys, z_phys])


def _extract_grid_whole(
    *,
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
    grid_params: dict[str, Any],
    smooth_sigma: float | None,
) -> np.ndarray:
    """Build a 3D density grid + marching-cubes shell for whole-kind grid extractors.

    Currently spherical only — for cylindrical droplets the cylinder
    axis would need to span the full simulation box, which the
    centered-grid convention doesn't accommodate naturally. Use a
    ``rays_*`` extractor for cylinder whole-fits.
    """
    if droplet_geometry.is_cylinder:
        raise NotImplementedError(
            "grid + whole-kind extraction is currently spherical-only. "
            "For cylinder droplets use InterfaceExtractor.rays_gaussian "
            "or rays_binning with surface_kind='whole'."
        )
    x_centers, y_centers, z_centers, density = _build_3d_density_grid(
        liquid_coordinates,
        center_geom,
        grid_params,
        smooth_sigma=smooth_sigma,
    )
    return _extract_isosurface_3d(
        x_centers, y_centers, z_centers, density, center_geom=center_geom
    )


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
        if surface_kind == "slicing":
            return _extract_grid_slicing(
                liquid_coordinates=liquid_coordinates,
                center_geom=center_geom,
                droplet_geometry=droplet_geometry,
                grid_params=self.grid_params,
                smooth_sigma=self.density_sigma,
            )
        return _extract_grid_whole(
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
        if surface_kind == "slicing":
            return _extract_grid_slicing(
                liquid_coordinates=liquid_coordinates,
                center_geom=center_geom,
                droplet_geometry=droplet_geometry,
                grid_params=self.grid_params,
                smooth_sigma=None,
            )
        return _extract_grid_whole(
            liquid_coordinates=liquid_coordinates,
            center_geom=center_geom,
            droplet_geometry=droplet_geometry,
            grid_params=self.grid_params,
            smooth_sigma=None,
        )
