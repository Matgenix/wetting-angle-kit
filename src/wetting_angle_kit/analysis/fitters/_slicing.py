"""Concrete slicing fitter: per-slice algebraic circle fits."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis.fitters._kasa import _kasa_circle_fit_2d
from wetting_angle_kit.analysis.fitters.base import (
    SlicingFitOutput,
    SurfaceFitter,
    SurfaceKind,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.wall import InterfaceData


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
