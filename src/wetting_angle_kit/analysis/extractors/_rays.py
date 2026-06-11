"""Ray-based extractor implementations + shared geometry/validation helpers."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis._density import (
    MIN_POINTS_PER_RAY,
    DensityFieldProtocol,
    GaussianDensityField,
    HistogramDensityField,
    fit_tanh_profiles_batched,
)
from wetting_angle_kit.analysis.extractors._sampling import (
    _fibonacci_sphere_directions,
)
from wetting_angle_kit.analysis.extractors.base import (
    InterfaceExtractor,
    SamplingKind,
    SurfaceKind,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.wall import InterfaceData


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
    azimuthal: float,
    max_dist: float,
    distances: np.ndarray,
    delta_polar: float,
) -> np.ndarray:
    """Per-slice ``(R, 2)`` interface from a tilted ray fan.

    Parameterised on a generic :class:`DensityFieldProtocol` so both
    ``rays_gaussian`` and ``rays_binning`` can share the geometry.
    """
    polar = np.linspace(0, 360, int(360 / delta_polar), endpoint=False)
    cos_polar = np.cos(np.deg2rad(polar))
    sin_polar = np.sin(np.deg2rad(polar))
    cos_azimuthal = np.cos(np.deg2rad(azimuthal))
    sin_azimuthal = np.sin(np.deg2rad(azimuthal))
    directions = np.column_stack(
        (cos_polar * cos_azimuthal, cos_polar * sin_azimuthal, sin_polar)
    )
    positions_rm = (
        center[None, None, :] + distances[None, :, None] * directions[:, None, :]
    )
    density_flat = field.evaluate(positions_rm.reshape(-1, 3))
    densities = density_flat.reshape(len(polar), len(distances))
    interface_re = fit_tanh_profiles_batched(distances, densities, max_dist=max_dist)
    x_proj = cos_polar * interface_re + center[0]
    z_proj = sin_polar * interface_re + center[2]
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
            azimuthals = np.linspace(0.0, 180.0, n_slices)
            return [
                _ray_slice_in_plane(
                    field, center_geom, float(g), max_dist, distances, delta_polar
                )
                for g in azimuthals
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
        directions = _fibonacci_sphere_directions(n_rays_sphere)
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
    polar = np.linspace(0, 360, int(360 / delta_polar), endpoint=False)
    cos_polar = np.cos(np.deg2rad(polar))
    sin_polar = np.sin(np.deg2rad(polar))
    cyl_directions = np.column_stack([cos_polar, np.zeros_like(polar), sin_polar])
    shells: list[np.ndarray] = []
    for y in ys:
        slice_center = np.array([center_geom[0], float(y), center_geom[2]])
        positions_rm = (
            slice_center[None, None, :]
            + distances[None, :, None] * cyl_directions[:, None, :]
        )
        density_flat = field.evaluate(positions_rm.reshape(-1, 3))
        densities = density_flat.reshape(len(polar), len(distances))
        interface_re = fit_tanh_profiles_batched(
            distances, densities, max_dist=max_dist
        )
        points = np.column_stack(
            [
                cos_polar * interface_re + slice_center[0],
                np.full(len(polar), float(y)),
                sin_polar * interface_re + slice_center[2],
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
