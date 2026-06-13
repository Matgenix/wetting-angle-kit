"""Concrete whole fitter: 3D sphere or 2D cylinder fit to the shell."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis.fitters._kasa import (
    _angle_from_cap,
    _whole_fit_one,
)
from wetting_angle_kit.analysis.fitters.base import (
    SurfaceFitter,
    SurfaceKind,
    WholeFitOutput,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.interface.base import InterfaceData


@dataclass(frozen=True, eq=False, kw_only=True)
class _WholeFitter(SurfaceFitter):
    """Concrete fitter for :meth:`SurfaceFitter.whole`."""

    kind: ClassVar[SurfaceKind] = "whole"

    surface_filter_offset: float
    bootstrap_samples: int

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be >= 0; got {self.bootstrap_samples}."
            )

    def validate_compatibility(
        self,
        droplet_geometry: DropletGeometry,  # noqa: ARG002 — ABC contract
    ) -> None:
        # Whole-fit covers spherical (sphere fit) and both cylinder
        # orientations (cylinder fit with the standard axis swap);
        # nothing geometry-specific to reject.
        return None

    def fit(
        self,
        interface_data: InterfaceData,
        z_wall: float,
        droplet_geometry: DropletGeometry,
    ) -> WholeFitOutput:
        if not isinstance(interface_data, np.ndarray):
            raise TypeError(
                "whole fitter expects an (N, 3) ndarray shell; "
                f"got {type(interface_data).__name__}."
            )
        if interface_data.ndim != 2 or interface_data.shape[1] != 3:
            raise ValueError(
                "whole fitter expects an (N, 3) ndarray shell; "
                f"got shape {interface_data.shape}."
            )

        z_filter = z_wall + self.surface_filter_offset
        kept = interface_data[interface_data[:, 2] > z_filter]
        spherical = droplet_geometry.is_spherical
        # Minimum points: 4 for a 3D sphere, 3 for a 2D circle.
        min_points = 4 if spherical else 3
        if len(kept) < min_points:
            raise RuntimeError(
                f"whole fit: only {len(kept)} shell points above "
                f"z_wall + surface_filter_offset = {z_filter:.3f} Å; "
                f"need at least {min_points} for the "
                f"{'sphere' if spherical else 'cylinder'} fit."
            )

        try:
            popt_shape, radius, zc = _whole_fit_one(kept, spherical=spherical)
        except (np.linalg.LinAlgError, ValueError) as e:
            raise RuntimeError(f"whole fit: geometric fit failed: {e}") from e

        angle = _angle_from_cap(z_wall, zc, radius)
        if angle is None:
            raise RuntimeError(
                f"whole fit: fitted shape (R={radius:.3f}, zc={zc:.3f}) "
                f"does not intersect wall plane z={z_wall:.3f}."
            )

        # Per-point residuals: distance to the fitted shape, in Å.
        if spherical:
            xc, yc, zc_fit, R_fit = popt_shape
            point_radius = np.sqrt(
                (kept[:, 0] - xc) ** 2
                + (kept[:, 1] - yc) ** 2
                + (kept[:, 2] - zc_fit) ** 2
            )
        else:
            xc, zc_fit, R_fit = popt_shape
            point_radius = np.hypot(kept[:, 0] - xc, kept[:, 2] - zc_fit)
        rms = float(np.sqrt(np.mean((point_radius - R_fit) ** 2)))

        angle_std = self._bootstrap_angle_std(kept, z_wall, spherical=spherical)

        # Pack popt with the wall position appended for plotting /
        # downstream reproduction. Spherical: [xc, yc, zc, R, z_wall].
        # Cylinder: [xc, zc, R, z_wall].
        popt = np.concatenate([popt_shape, [z_wall]])

        return WholeFitOutput(
            angle=angle,
            z_wall=z_wall,
            rms_residual=rms,
            angle_std=angle_std,
            interface_shell=kept,
            popt=popt,
        )

    def _bootstrap_angle_std(
        self, kept: np.ndarray, z_wall: float, *, spherical: bool
    ) -> float | None:
        if self.bootstrap_samples <= 0:
            return None
        # Deterministic seed so result is reproducible per (analyzer, batch).
        rng = np.random.default_rng(0)
        n = len(kept)
        bootstrap_angles: list[float] = []
        for _ in range(self.bootstrap_samples):
            idx = rng.integers(0, n, n)
            sample = kept[idx]
            try:
                _, b_R, b_zc = _whole_fit_one(sample, spherical=spherical)
            except (np.linalg.LinAlgError, ValueError):
                continue
            a = _angle_from_cap(z_wall, b_zc, b_R)
            if a is not None:
                bootstrap_angles.append(a)
        if not bootstrap_angles:
            return None
        return float(np.std(bootstrap_angles))
