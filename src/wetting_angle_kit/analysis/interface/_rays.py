"""Ray-based sampling: ``_RaysSampling`` + ray-fan geometry helpers."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis._density import (
    MIN_POINTS_PER_RAY,
    DensityFieldProtocol,
    fit_tanh_profiles_batched,
)
from wetting_angle_kit.analysis.density_estimator import DensityEstimator
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface.base import (
    InterfaceData,
    SamplingKind,
    SpaceSampling,
    SurfaceKind,
)


def _fibonacci_sphere_directions(n: int) -> np.ndarray:
    """Equal-area Fibonacci-spiral directions on the full sphere.

    ``cos θ`` is uniformly spaced over ``[-1, 1]`` (so the surface
    density is uniform over the whole sphere) and ``φ`` is incremented
    by the golden angle for low-discrepancy azimuthal coverage.
    ``i = 0`` sits at the south pole (``cos θ = -1``) and
    ``i = n - 1`` at the north pole (``cos θ = 1``).

    The full sphere coverage is important for sessile droplets: rays
    emitted from the droplet COM in downward directions traverse the
    liquid, hit the wall plane, and contribute interface points at the
    wall — making :meth:`WallDetector.min_plus_offset` work correctly
    in the whole-fit pipeline. (Restricting to the upper hemisphere
    misses the wall, so ``min(shell z)`` lands on ``COM_z`` instead.)
    """
    if n <= 0:
        return np.empty((0, 3))
    i = np.arange(n, dtype=np.float64)
    cos_theta = 2.0 * i / (n - 1) - 1.0 if n > 1 else np.array([1.0])
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

    Both density estimators share the same ray-fan
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
    distances: np.ndarray,
    delta_polar: float,
) -> np.ndarray:
    """Per-slice ``(R, 2)`` interface from a tilted ray fan.

    Parameterised on a generic :class:`DensityFieldProtocol` so both
    the Gaussian and binning density paths can share the geometry.
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
    interface_re = fit_tanh_profiles_batched(distances, densities)
    # Rays with no resolvable interface return NaN; drop them rather
    # than seeding a spurious point.
    resolved = np.isfinite(interface_re)
    x_proj = cos_polar[resolved] * interface_re[resolved] + center[0]
    z_proj = sin_polar[resolved] * interface_re[resolved] + center[2]
    return np.column_stack([x_proj, z_proj])


def _compute_max_dist(
    liquid_coordinates: np.ndarray,
    center_geom: np.ndarray,
    *,
    buffer: float = 10.0,
) -> float:
    """Ray-sampling envelope derived from atom positions.

    Returns the maximum distance from any atom to ``center_geom`` plus
    a small ``buffer`` (Å) so the tanh fit has headroom past the
    interface (a few smoothing-σ worth, for typical density_sigma ≈ 3).
    Defaults to ``buffer`` alone when the atom set is empty.
    """
    if liquid_coordinates.size == 0:
        return float(buffer)
    distances = np.linalg.norm(liquid_coordinates - center_geom[None, :], axis=1)
    return float(np.max(distances) + buffer)


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

    The density evaluator is provided by the caller; the geometry,
    sampling cadence, and tanh-fit invocation are shared across all
    density estimators.
    """
    n_samples = max(int(max_dist * points_per_angstrom), MIN_POINTS_PER_RAY)
    distances = np.linspace(0.0, max_dist, n_samples)

    if surface_kind == "slicing":
        if droplet_geometry.is_spherical:
            assert delta_azimuthal is not None
            n_slices = int(180 / delta_azimuthal)
            azimuthals = np.linspace(0.0, 180.0, n_slices, endpoint=False)
            return [
                _ray_slice_in_plane(
                    field, center_geom, float(g), distances, delta_polar
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
                _ray_slice_in_plane(field, slice_center, 0.0, distances, delta_polar)
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
        interface_re = fit_tanh_profiles_batched(distances, densities)
        # Drop rays with no resolvable interface (NaN).
        resolved = np.isfinite(interface_re)
        return (
            center_geom[None, :] + interface_re[resolved, None] * directions[resolved]
        )

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
        interface_re = fit_tanh_profiles_batched(distances, densities)
        # Drop rays with no resolvable interface (NaN).
        resolved = np.isfinite(interface_re)
        points = np.column_stack(
            [
                cos_polar[resolved] * interface_re[resolved] + slice_center[0],
                np.full(int(resolved.sum()), float(y)),
                sin_polar[resolved] * interface_re[resolved] + slice_center[2],
            ]
        )
        shells.append(points)
    return np.concatenate(shells, axis=0) if shells else np.empty((0, 3))


@dataclass(frozen=True, eq=False, kw_only=True)
class _RaysSampling(SpaceSampling):
    """Concrete sampling for :meth:`SpaceSampling.rays`."""

    kind: ClassVar[SamplingKind] = "rays"

    delta_azimuthal: float | None
    delta_cylinder: float | None
    n_rays_sphere: int | None
    delta_polar: float
    points_per_angstrom: float

    def validate_compatibility(
        self,
        surface_kind: SurfaceKind,
        droplet_geometry: DropletGeometry,
    ) -> None:
        _validate_rays_params(
            name="rays",
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
        surface_kind: SurfaceKind,
        density: DensityEstimator,
    ) -> InterfaceData:
        field = density.build_field(liquid_coordinates)
        max_dist = _compute_max_dist(liquid_coordinates, center_geom)
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
