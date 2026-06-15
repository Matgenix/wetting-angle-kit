"""Grid-based extractor implementation.

This sampling evaluates a density field at fixed-cell grid points and
traces the half-bulk iso-contour (slicing mode) or iso-surface (whole
mode). For slicing mode, it iterates per-slice — azimuthal angles for
spherical droplets, axial steps for cylindrical droplets — so the
downstream :class:`SurfaceFitter.slicing` sees one ``(s, z)`` contour
per slice and can report per-slice scatter.

The per-cell density comes from the :class:`DensityEstimator`
strategy passed via :meth:`SpaceSampling.grid` × :class:`DensityEstimator`. The Gaussian
variant samples the KDE at cell centres; the binning variant
histograms atoms into cells (with a slab cut perpendicular to the
slice plane for slicing mode).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from wetting_angle_kit.analysis._grid_utils import edges_from_bin_width
from wetting_angle_kit.analysis.density_estimator import (
    DensityEstimator,
    _GaussianDensityEstimator,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface.base import (
    InterfaceData,
    SamplingKind,
    SpaceSampling,
    SurfaceKind,
)

_GRID_KEYS_2D = frozenset(
    {"xi_0", "xi_f", "bin_width_x", "zi_0", "zi_f", "bin_width_z"}
)
_GRID_KEYS_3D = _GRID_KEYS_2D | {"yi_0", "yi_f", "bin_width_y"}

#: Default cell width for the grid + binning combination (Å). The
#: histogram estimator has no smoothing scale to anchor to, so a
#: flat default is used.
#:
#: 2 Å is the compromise for *pooled-batch* analyses; for
#: per-frame slicing-mode use the slab cut leaves too few atoms
#: per cell regardless of ``bin_width``, and the user should either
#: pool multiple frames per batch or supply a hand-tuned
#: ``grid_params`` explicitly.
_DEFAULT_BIN_WIDTH_BINNING = 2.0

#: Buffer (Å) added to the atom bounding box when auto-deriving grid
#: range bounds. Matches the buffer used by ``_compute_max_dist`` for
#: the ray extractors, keeping the spatial-envelope rule consistent.
_DEFAULT_GRID_BUFFER = 5.0


def _default_grid_params(
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
    *,
    surface_kind: SurfaceKind,
    bin_width: float,
    buffer: float = _DEFAULT_GRID_BUFFER,
) -> dict[str, Any]:
    """Atom-derived default ``grid_params``.

    Range bounds come from the atom bounding box (in the droplet-centred
    frame for spherical, lab-frame for the cylinder axis) plus a
    fixed buffer. Cell widths are uniform across the three axes and
    set by the caller — typically ``density_sigma / 2`` for the
    Gaussian KDE estimator or :data:`_DEFAULT_BIN_WIDTH_BINNING` for
    the histogram.
    """
    if liquid_coordinates.size == 0:
        # Empty batch: degenerate grid, the iso-contour will be empty.
        return (
            {
                "xi_0": -buffer,
                "xi_f": buffer,
                "bin_width_x": bin_width,
                "zi_0": 0.0,
                "zi_f": buffer,
                "bin_width_z": bin_width,
            }
            if surface_kind == "slicing"
            else {
                "xi_0": -buffer,
                "xi_f": buffer,
                "bin_width_x": bin_width,
                "yi_0": -buffer,
                "yi_f": buffer,
                "bin_width_y": bin_width,
                "zi_0": 0.0,
                "zi_f": buffer,
                "bin_width_z": bin_width,
            }
        )
    # Atom extent. For slicing-spherical, the slice plane's ``s`` axis
    # is the radial direction in ``(x, y)``, so the natural envelope is
    # ``max(hypot(dx, dy))``. For slicing-cylinder, the slice plane's
    # ``s`` axis is purely ``x`` (the in-plane direction perpendicular
    # to the cylinder axis ``y``), so only the radial x-extent matters
    # — using ``hypot`` would oversize the grid with the cylinder
    # length contribution.
    dx = liquid_coordinates[:, 0] - float(center_geom[0])
    dy = liquid_coordinates[:, 1] - float(center_geom[1])
    z_max = float(liquid_coordinates[:, 2].max()) + buffer
    if surface_kind == "slicing":
        if droplet_geometry.is_spherical:
            in_plane_max = float(np.max(np.hypot(dx, dy))) + buffer
        else:
            in_plane_max = float(np.max(np.abs(dx))) + buffer
        return {
            "xi_0": -in_plane_max,
            "xi_f": in_plane_max,
            "bin_width_x": bin_width,
            "zi_0": 0.0,
            "zi_f": z_max,
            "bin_width_z": bin_width,
        }
    # Whole-mode 3D grid. For cylindrical droplets the ``y`` axis is
    # the cylinder axis and atoms span the full box; the bounding box
    # (with buffer) captures that.
    y_min = float(dy.min()) - buffer
    y_max = float(dy.max()) + buffer
    x_max = float(np.max(np.abs(dx))) + buffer
    return {
        "xi_0": -x_max,
        "xi_f": x_max,
        "bin_width_x": bin_width,
        "yi_0": y_min,
        "yi_f": y_max,
        "bin_width_y": bin_width,
        "zi_0": 0.0,
        "zi_f": z_max,
        "bin_width_z": bin_width,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_grid_params(
    *,
    name: str,
    grid_params: dict[str, Any],
    surface_kind: SurfaceKind,
) -> None:
    """Check ``grid_params`` carries the right keys + scikit-image for whole-mode."""
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


def _validate_per_slice_params(
    *,
    name: str,
    delta_azimuthal: float | None,
    delta_cylinder: float | None,
    droplet_geometry: DropletGeometry,
) -> None:
    """For slicing-mode grid extractors: require the right slice-step param."""
    if droplet_geometry.is_spherical and delta_azimuthal is None:
        raise ValueError(f"{name} for slicing+spherical requires delta_azimuthal.")
    if droplet_geometry.is_cylinder and delta_cylinder is None:
        raise ValueError(
            f"{name} for slicing+{droplet_geometry.name} requires delta_cylinder."
        )


# ---------------------------------------------------------------------------
# Edge helpers (cell-width-based)
# ---------------------------------------------------------------------------


def _slice_grid_edges(
    grid_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    s_edges = edges_from_bin_width(
        grid_params["xi_0"], grid_params["xi_f"], grid_params["bin_width_x"]
    )
    z_edges = edges_from_bin_width(
        grid_params["zi_0"], grid_params["zi_f"], grid_params["bin_width_z"]
    )
    return s_edges, z_edges


def _slice_grid_centres(
    grid_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    s_edges, z_edges = _slice_grid_edges(grid_params)
    return (
        0.5 * (s_edges[:-1] + s_edges[1:]),
        0.5 * (z_edges[:-1] + z_edges[1:]),
    )


def _whole_grid_edges(
    grid_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = edges_from_bin_width(
        grid_params["xi_0"], grid_params["xi_f"], grid_params["bin_width_x"]
    )
    y_edges = edges_from_bin_width(
        grid_params["yi_0"], grid_params["yi_f"], grid_params["bin_width_y"]
    )
    z_edges = edges_from_bin_width(
        grid_params["zi_0"], grid_params["zi_f"], grid_params["bin_width_z"]
    )
    return x_edges, y_edges, z_edges


def _whole_grid_centres(
    grid_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges, y_edges, z_edges = _whole_grid_edges(grid_params)
    return (
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:]),
        0.5 * (z_edges[:-1] + z_edges[1:]),
    )


# ---------------------------------------------------------------------------
# Slice iteration
# ---------------------------------------------------------------------------


def _iter_slice_planes(
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    droplet_geometry: DropletGeometry,
    *,
    delta_azimuthal: float | None,
    delta_cylinder: float | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(slice_center, in_plane_axis)`` for each slicing plane.

    ``in_plane_axis`` is the unit vector defining the in-plane radial
    coordinate ``s``. The perpendicular-to-plane axis is recovered as
    ``(-in_plane_axis[1], in_plane_axis[0], 0)`` by the binning
    estimator's slab cut.
    """
    if droplet_geometry.is_spherical:
        assert delta_azimuthal is not None
        n_slices = int(180 / delta_azimuthal)
        for gamma_deg in np.linspace(0.0, 180.0, n_slices, endpoint=False):
            gamma_rad = float(np.deg2rad(gamma_deg))
            in_plane = np.array([np.cos(gamma_rad), np.sin(gamma_rad), 0.0])
            yield np.asarray(center_geom, dtype=float), in_plane
        return
    # cylinder
    assert delta_cylinder is not None
    y_vals = liquid_coordinates[:, 1]
    ys = np.arange(float(y_vals.min()), float(y_vals.max()), delta_cylinder)
    in_plane = np.array([1.0, 0.0, 0.0])
    for y in ys:
        slice_center = np.array(
            [float(center_geom[0]), float(y), float(center_geom[2])]
        )
        yield slice_center, in_plane


# ---------------------------------------------------------------------------
# Iso-contour / iso-surface extraction
# ---------------------------------------------------------------------------


def _extract_isocontour_2d(
    s_centers: np.ndarray,
    z_centers: np.ndarray,
    density: np.ndarray,
    *,
    fraction_of_bulk: float = 0.5,
    bulk_percentile: float = 95.0,
) -> np.ndarray:
    """Longest density iso-line as ``(M, 2)`` ``(s, z)`` points."""
    from skimage.measure import find_contours

    if density.size == 0 or float(density.max()) <= 0:
        return np.empty((0, 2))
    bulk = float(np.percentile(density, bulk_percentile))
    if bulk <= 0:
        return np.empty((0, 2))
    level = fraction_of_bulk * bulk
    contours = find_contours(density, level)  # type: ignore[no-untyped-call,unused-ignore]
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)
    ds = (float(s_centers[-1]) - float(s_centers[0])) / max(len(s_centers) - 1, 1)
    dz = (float(z_centers[-1]) - float(z_centers[0])) / max(len(z_centers) - 1, 1)
    s_phys = float(s_centers[0]) + ds * longest[:, 0]
    z_phys = float(z_centers[0]) + dz * longest[:, 1]
    return np.column_stack([s_phys, z_phys])


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
    """Marching-cubes shell shifted back to absolute lab coords."""
    from skimage.measure import marching_cubes

    if density.size == 0 or float(density.max()) <= 0:
        return np.empty((0, 3))
    bulk = float(np.percentile(density, bulk_percentile))
    if bulk <= 0:
        return np.empty((0, 3))
    level = fraction_of_bulk * bulk
    try:
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


# ---------------------------------------------------------------------------
# Extractor classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, kw_only=True)
class _GridSampling(SpaceSampling):
    """Concrete sampling for :meth:`SpaceSampling.grid`.

    Dispatches the per-cell density computation to the
    :class:`DensityEstimator` strategy received at extract time: the
    Gaussian variant samples the KDE at cell centres, the binning
    variant histograms atoms into cells (with a slab cut for slicing
    mode).
    """

    kind: ClassVar[SamplingKind] = "grid"

    grid_params: dict[str, Any] | None
    delta_azimuthal: float | None
    delta_cylinder: float | None

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        # Key-presence check is skipped when grid_params is None
        # (auto-derived in extract); the scikit-image import check for
        # whole-mode still runs so the user gets the error at
        # construction.
        if self.grid_params is not None:
            _validate_grid_params(
                name="grid",
                grid_params=self.grid_params,
                surface_kind=surface_kind,
            )
        elif surface_kind == "whole":
            try:
                import skimage.measure  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "grid for whole-kind extraction requires "
                    "scikit-image (used for marching_cubes). Install with: "
                    "pip install 'wetting-angle-kit[grid3d]'."
                ) from e
        if surface_kind == "slicing":
            _validate_per_slice_params(
                name="grid",
                delta_azimuthal=self.delta_azimuthal,
                delta_cylinder=self.delta_cylinder,
                droplet_geometry=droplet_geometry,
            )

    def _auto_grid_bin_width(self, density: DensityEstimator) -> float:
        # Pick the auto-derived bin_width that matches the estimator:
        # Gaussian uses density_sigma / 2 (Nyquist-ish for the KDE);
        # histograms use a flat 2 Å (no smoothing scale to anchor to).
        if isinstance(density, _GaussianDensityEstimator):
            return density.density_sigma / 2.0
        return _DEFAULT_BIN_WIDTH_BINNING

    def extract(
        self,
        liquid_coordinates: np.ndarray,
        center_geom: np.ndarray,
        droplet_geometry: DropletGeometry,
        surface_kind: SurfaceKind,
        density: DensityEstimator,
    ) -> InterfaceData:
        grid_params = self.grid_params or _default_grid_params(
            liquid_coordinates,
            center_geom,
            droplet_geometry,
            surface_kind=surface_kind,
            bin_width=self._auto_grid_bin_width(density),
        )
        if surface_kind == "slicing":
            s_centers, z_centers = _slice_grid_centres(grid_params)
            # Slab thickness perpendicular to the slice plane equals
            # the in-plane horizontal cell width, so each cell's bin
            # is a ``ds × dz × ds`` box (square cross-section in the
            # ``(s, perp)`` plane). The Gaussian estimator ignores
            # this parameter — its kernel size is set by
            # ``density_sigma`` on the estimator itself.
            s_edges, _z_edges = _slice_grid_edges(grid_params)
            slab = float(s_edges[1] - s_edges[0])
            contours: list[np.ndarray] = []
            for slice_center, in_plane_axis in _iter_slice_planes(
                liquid_coordinates,
                center_geom,
                droplet_geometry,
                delta_azimuthal=self.delta_azimuthal,
                delta_cylinder=self.delta_cylinder,
            ):
                slice_density = density.evaluate_on_slice(
                    liquid_coordinates,
                    slice_center,
                    in_plane_axis,
                    s_centers,
                    z_centers,
                    slab,
                )
                contours.append(
                    _extract_isocontour_2d(s_centers, z_centers, slice_density)
                )
            return contours
        x_centers, y_centers, z_centers = _whole_grid_centres(grid_params)
        density3d = density.evaluate_on_3d_grid(
            liquid_coordinates,
            x_centers,
            y_centers,
            z_centers,
            x_offset=float(center_geom[0]),
            y_offset=float(center_geom[1]),
        )
        return _extract_isosurface_3d(
            x_centers,
            y_centers,
            z_centers,
            density3d,
            center_geom=center_geom,
        )
