"""Concrete slicing fitter: per-slice algebraic circle fits."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis.fitters._taubin import _taubin_circle_fit_2d
from wetting_angle_kit.analysis.fitters.base import (
    SlicingFitOutput,
    SurfaceFitter,
    SurfaceKind,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface.base import InterfaceData


@dataclass(frozen=True, eq=False, kw_only=True)
class _SlicingFitter(SurfaceFitter):
    """Concrete fitter for :meth:`SurfaceFitter.slicing`."""

    kind: ClassVar[SurfaceKind] = "slicing"

    surface_filter_offset: float

    def validate_compatibility(
        self,
        droplet_geometry: DropletGeometry,  # noqa: ARG002 — ABC contract
    ) -> None:
        # Slicing handles all three geometries (spherical and both
        # cylinder orientations); nothing geometry-specific to reject.
        return None

    def fit(
        self,
        interface_data: InterfaceData,
        z_wall: float,
        droplet_geometry: DropletGeometry,  # noqa: ARG002 — ABC contract
    ) -> SlicingFitOutput:
        if not isinstance(interface_data, list):
            raise TypeError(
                "slicing fitter expects a list of per-slice (M, 2) arrays; "
                f"got {type(interface_data).__name__}."
            )

        z_filter = z_wall + self.surface_filter_offset
        # Per-slice arrays stay full length and index-aligned: a slice
        # that yields no valid angle is recorded as NaN rather than
        # dropped, so attrition is visible and the slice index is kept.
        per_slice_angles: list[float] = []
        slice_surfaces: list[np.ndarray] = []
        slice_popts: list[np.ndarray] = []
        slice_rms_residuals: list[float] = []
        nan_popt: np.ndarray = np.full(4, np.nan)

        for surf in interface_data:
            slice_surfaces.append(surf)
            angle, rms, popt = float("nan"), float("nan"), nan_popt
            # Need at least 3 non-collinear points to fit a circle.
            kept = surf[surf[:, 1] > z_filter] if surf.size else surf
            if len(kept) >= 3:
                try:
                    xc, zc, radius = _taubin_circle_fit_2d(kept[:, 0], kept[:, 1])
                    # Contact angle from circle / wall-line intersection:
                    # ``cos θ = (z_wall - z_center) / R``. A circle that
                    # doesn't reach the wall (``|Δz| ≥ R``) yields no angle.
                    delta_z = z_wall - zc
                    if abs(delta_z) < radius:
                        angle = float(np.degrees(np.arccos(delta_z / radius)))
                        # Per-slice RMS of the circle-fit residuals (Å).
                        radii = np.hypot(kept[:, 0] - xc, kept[:, 1] - zc)
                        rms = float(np.sqrt(np.mean((radii - radius) ** 2)))
                        popt = np.array([xc, zc, radius, z_wall])
                except (np.linalg.LinAlgError, ValueError):
                    pass
            per_slice_angles.append(angle)
            slice_rms_residuals.append(rms)
            slice_popts.append(popt)

        angles_arr = np.asarray(per_slice_angles, dtype=float)
        n_slices_total = len(angles_arr)
        n_slices_used = int(np.isfinite(angles_arr).sum())
        if n_slices_used == 0:
            raise RuntimeError(
                "slicing fit: no slice produced a valid contact angle "
                f"after filtering and circle fitting ({n_slices_total} "
                "slice(s) attempted)."
            )

        # nanmean / nanstd: the batch angle and its spread are over the
        # slices that produced a value; the NaN entries above keep the
        # dropped slices visible.
        return SlicingFitOutput(
            angle=float(np.nanmean(angles_arr)),
            z_wall=z_wall,
            rms_residual=float(np.nanmean(slice_rms_residuals)),
            angle_std=float(np.nanstd(angles_arr)),
            per_slice_angles=angles_arr,
            slice_surfaces=slice_surfaces,
            slice_popts=np.asarray(slice_popts, dtype=float),
            n_slices_total=n_slices_total,
            n_slices_used=n_slices_used,
        )
