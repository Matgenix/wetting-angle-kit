from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SlicingResults:
    """In-memory container for the per-frame output of the slicing method.

    Replaces the legacy ``all_angles.npy`` / ``all_surfaces.npy`` /
    ``all_popts.npy`` round-trip. The three parallel lists share the same
    indexing as ``frames``: entry ``i`` describes frame ``frames[i]``.

    Attributes
    ----------
    frames : list[int]
        Frame indices that were successfully processed, sorted ascending.
    angles : list[np.ndarray]
        Per-frame array of contact angles (one value per slice).
    surfaces : list[list[np.ndarray]]
        Per-frame list of slice surface contours; each contour is an
        ``(N, 2)`` array of ``(x, z)`` vertex coordinates.
    popts : list[np.ndarray]
        Per-frame array of fitted circle parameters; each entry has shape
        ``(n_slices, 4)`` with columns ``(x_center, z_center, radius, extra)``.
    method_metadata : dict
        Free-form method descriptor (e.g. ``{"frames_per_angle": 1}``).
    """

    frames: list[int]
    angles: list[np.ndarray]
    surfaces: list[list[np.ndarray]]
    popts: list[np.ndarray]
    method_metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def per_frame_mean_angles(self) -> np.ndarray:
        """Per-frame mean contact angle, taken across slices, in degrees."""
        return np.array([float(np.mean(a)) for a in self.angles])

    @property
    def mean_angle(self) -> float:
        """Mean contact angle across frames, in degrees."""
        return float(np.mean(self.per_frame_mean_angles))

    @property
    def std_angle(self) -> float:
        """Standard deviation of the per-frame mean contact angle, in degrees."""
        return float(np.std(self.per_frame_mean_angles))
