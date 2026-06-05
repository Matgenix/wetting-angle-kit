from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BinningBatch:
    """Per-batch output of the binning analysis.

    A batch is the fitting unit: a contiguous group of frames whose
    coordinates are aggregated into a single 2D density field that is then
    fitted to extract one contact angle.

    Attributes
    ----------
    batch_index : int
        Sequential identifier (starting at 1) for the batch.
    angle : float
        Fitted contact angle in degrees (``nan`` if the fit failed).
    n_particles : float
        Average number of fluid particles per frame within the batch.
    xi_cc : np.ndarray
        Cell-center coordinates along the radial/in-plane axis (1D).
    zi_cc : np.ndarray
        Cell-center coordinates along the vertical axis (1D).
    rho_cc : np.ndarray
        2D density field on the ``xi_cc × zi_cc`` grid (particles · Å⁻³).
    circle_xi : np.ndarray | None
        Fitted droplet circle iso-line, radial coordinates. ``None`` when
        :meth:`HyperbolicTangentModel.compute_isoline` failed (non-physical
        fit).
    circle_zi : np.ndarray | None
        Fitted droplet circle iso-line, vertical coordinates.
    wall_line_xi : np.ndarray | None
        Fitted wall position, radial coordinates.
    wall_line_zi : np.ndarray | None
        Fitted wall position, vertical coordinates.
    fitted_params : dict[str, float]
        Fitted model parameters (e.g. ``R_eq``, ``zi_c``, ``zi_0``).
    """

    batch_index: int
    angle: float
    n_particles: float
    xi_cc: np.ndarray
    zi_cc: np.ndarray
    rho_cc: np.ndarray
    circle_xi: np.ndarray | None
    circle_zi: np.ndarray | None
    wall_line_xi: np.ndarray | None
    wall_line_zi: np.ndarray | None
    fitted_params: dict[str, float] = field(default_factory=dict)


@dataclass
class BinningResults:
    """In-memory container for the binning method output.

    Replaces the legacy ``log_data_batch_*.txt`` / ``rho_field_batch_*.csv``
    round-trip: every quantity needed downstream (statistics, contour plot,
    per-batch angle evolution) is carried as attributes on the batches.

    Attributes
    ----------
    batches : list[BinningBatch]
        One entry per fitted batch, in batch order.
    method_metadata : dict
        Free-form method descriptor (e.g. ``{"frames_per_trajectory": 100}``).
    """

    batches: list[BinningBatch]
    method_metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def angles_per_batch(self) -> np.ndarray:
        """Per-batch fitted contact angle, in degrees."""
        return np.array([b.angle for b in self.batches])

    @property
    def mean_angle(self) -> float:
        """Mean contact angle across batches, in degrees."""
        return float(np.mean(self.angles_per_batch))

    @property
    def std_angle(self) -> float:
        """Standard deviation of the per-batch contact angle, in degrees."""
        return float(np.std(self.angles_per_batch))
