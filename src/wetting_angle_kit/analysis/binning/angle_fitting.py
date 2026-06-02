"""Binning-method contact-angle analyzer.

Algorithm
---------

The trajectory is aggregated into a 2D density field ``rho(xi, zi)`` on a
regular bin grid, where ``xi`` is the in-plane radial coordinate produced
by :func:`project_to_profile` and ``zi`` is the lab-frame vertical
coordinate. The histogram uses :func:`numpy.histogram2d` (left-edge
inclusive, right-edge exclusive,
last bin closed on both ends).

Per-bin volume elements:

* ``cylinder_x`` / ``cylinder_y``: ``dV = 2 * box_dimension * dxi * dzi``,
  where ``box_dimension`` is the box length along the cylinder axis read
  from the parser. The factor of 2 accounts for folding the symmetric
  distribution into positive ``xi`` via ``|x_centered|``.
* ``spherical``: ``dV = 2 * pi * xi_cc * dxi * dzi`` — the annular shell
  volume of cylindrical coordinates.

A :class:`HyperbolicTangentModel` is then fitted to the time-averaged
density field and the implied contact angle is derived from the fitted
sphere radius, center, and wall position. Lengths are in Å, densities in
particles · Å⁻³, and the final contact angle is returned in degrees.
"""

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from wetting_angle_kit.analysis.binning.results import BinningBatch
from wetting_angle_kit.analysis.binning.surface_definition import (
    HyperbolicTangentModel,
)
from wetting_angle_kit.io_utils import (
    project_to_profile,
    validate_droplet_geometry,
)

logger = logging.getLogger(__name__)

_PARAM_NAMES = ("rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2")


class BinningBatchFitter:
    """Binning-based contact angle estimator using density field fitting.

    Frames aggregated in spatial bins form a time-averaged density field.
    A hyperbolic tangent interface model is fitted and the implied contact
    angle is computed from fitted geometric parameters.
    """

    def __init__(
        self,
        parser: Any,
        atom_indices: Any,
        droplet_geometry: str = "spherical",
        binning_params: dict[str, Any] | None = None,
        precentered: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser providing coordinates and box dimensions.
        atom_indices : Any
            Indices (or IDs) of liquid atoms to include in the density field.
        droplet_geometry : str, default "spherical"
            One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.
        binning_params : dict, optional
            Grid definition with keys ``xi_0``, ``xi_f``, ``nbins_xi``,
            ``zi_0``, ``zi_f``, ``nbins_zi``. A heuristic default is used if None.
        precentered : bool, default False
            Set True to declare that the trajectory already recenters the
            droplet at every frame and atoms are not wrapped across periodic
            boundaries. The per-frame circular-mean recentering is then
            skipped (using a plain arithmetic mean instead), removing the
            associated overhead. Setting this on a trajectory that does NOT
            satisfy the precondition will produce wrong results.
        """
        validate_droplet_geometry(droplet_geometry)
        self.parser = parser
        self.atom_indices = atom_indices
        self.droplet_geometry = droplet_geometry
        self.precentered = precentered
        if binning_params is None:
            max_dist = int(
                np.max(
                    np.array(
                        [
                            parser.box_size_y(frame_index=0),
                            parser.box_size_x(frame_index=0),
                        ]
                    )
                )
                / 3
            )
            self.binning_params = {
                "xi_0": 0,
                "xi_f": max_dist,
                "nbins_xi": 50,
                "zi_0": 0.0,
                "zi_f": max_dist,
                "nbins_zi": 50,
            }
            warnings.warn(
                "binning_params was not supplied; using a heuristic default "
                f"(xi_0=0, xi_f={max_dist}, zi_0=0, zi_f={max_dist}, "
                "50x50 bins) derived from one third of the largest in-plane "
                "box dimension. For accurate density fields, supply "
                "system-specific binning_params matching your droplet size "
                "and per-frame sampling.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.binning_params = binning_params
        self._initialize_grid()
        if self.droplet_geometry == "cylinder_x":
            self.box_dimension = self.parser.box_size_x(frame_index=0)
        elif self.droplet_geometry == "cylinder_y":
            self.box_dimension = self.parser.box_size_y(frame_index=0)
        else:
            self.box_dimension = None

    def _initialize_grid(self) -> None:
        """Initialize bin edges, centers and cell sizes from parameters."""
        self.xi = np.linspace(
            self.binning_params["xi_0"],
            self.binning_params["xi_f"],
            int(self.binning_params["nbins_xi"]),
        )
        self.zi = np.linspace(
            self.binning_params["zi_0"],
            self.binning_params["zi_f"],
            int(self.binning_params["nbins_zi"]),
        )
        self.dxi = self.xi[1] - self.xi[0]
        self.dzi = self.zi[1] - self.zi[0]
        self.xi_cc = 0.5 * (self.xi[1:] + self.xi[:-1])
        self.zi_cc = 0.5 * (self.zi[1:] + self.zi[:-1])

    def get_profile_coordinates(
        self,
        frame_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute 2D projection coordinates (r, z) for contact angle analysis.

        Projects 3D atomic positions onto a 2D plane based on the assumed
        droplet geometry. Coordinates are accumulated across all requested
        frames in lockstep.

        Parameters
        ----------
        frame_indices : Sequence[int]
            Frame indices to process.

        Returns
        -------
        r_values : ndarray
            Concatenated radial distances.
        z_values : ndarray
            Concatenated vertical coordinates.
        n_frames : int
            Number of frames processed (``len(frame_indices)``).
        """
        validate_droplet_geometry(self.droplet_geometry)
        r_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        # ``precentered=True`` skips the box probe and uses arithmetic-mean
        # centering; otherwise box_size is queried per-frame for PBC-aware
        # recentering. The parser ABC enforces box_size_x/y, so no fallback
        # is needed.
        box_size: tuple[float, float] | None = None
        if frame_indices and not self.precentered:
            box_size = (
                self.parser.box_size_x(frame_index=frame_indices[0]),
                self.parser.box_size_y(frame_index=frame_indices[0]),
            )
        for frame_idx in frame_indices:
            positions = self.parser.parse(frame_idx, self.atom_indices)
            if box_size is not None:
                box_size = (
                    self.parser.box_size_x(frame_index=frame_idx),
                    self.parser.box_size_y(frame_index=frame_idx),
                )
            r_frame, z_frame = project_to_profile(
                positions, self.droplet_geometry, box_size=box_size
            )
            r_chunks.append(r_frame)
            z_chunks.append(z_frame)
            if frame_idx % 10 == 0:
                x_cm = (
                    np.mean(positions, axis=0) if positions.size else np.full(3, np.nan)
                )
                logger.info(
                    f"Frame {frame_idx}: {len(positions)} particles, "
                    f"center of mass {np.array2string(x_cm, precision=3)}"
                )
        r_values = np.concatenate(r_chunks) if r_chunks else np.empty(0)
        z_values = np.concatenate(z_chunks) if z_chunks else np.empty(0)
        if r_values.size > 0:
            logger.info(
                f"r range: ({float(r_values.min()):.3f}, {float(r_values.max()):.3f})"
            )
            logger.info(
                f"z range: ({float(z_values.min()):.3f}, {float(z_values.max()):.3f})"
            )
        return r_values, z_values, len(frame_indices)

    def binning(
        self, xi_par: np.ndarray, zi_par: np.ndarray, len_frames: int
    ) -> np.ndarray:
        """Return 2D density field by binning particle coordinates.

        Uses :func:`numpy.histogram2d`, which is vectorized (O(N) in the
        particle count) and correctly handles particles on bin edges
        (inclusive on the left/lower edge, inclusive on the right/upper
        edge of the last bin only). This makes the legacy ``+0.01`` shift
        on the radial coordinate unnecessary.

        Parameters
        ----------
        xi_par : ndarray
            Radial/in-plane coordinate values for particles over frames.
        zi_par : ndarray
            Vertical coordinate values for particles over frames.
        len_frames : int
            Number of frames aggregated.

        Returns
        -------
        ndarray, shape (nbins_xi-1, nbins_zi-1)
            Averaged density field on cell centers.
        """
        counts, _, _ = np.histogram2d(
            xi_par,
            zi_par,
            bins=(self.xi, self.zi),
        )
        if self.droplet_geometry in ("cylinder_x", "cylinder_y"):
            dV = 2.0 * self.box_dimension * self.dxi * self.dzi
            rho_cc = counts / dV
        else:  # spherical droplet geometry
            dV_per_row = 2.0 * np.pi * self.xi_cc * self.dxi * self.dzi
            rho_cc = counts / dV_per_row[:, np.newaxis]
        if len_frames > 0:
            rho_cc /= len_frames
        return rho_cc

    def process_batch(
        self,
        frame_list: list[int],
        model: Any | None = None,
        batch_index: int | None = None,
    ) -> BinningBatch:
        """Process a batch of frames and return its fitted contact-angle data.

        Parameters
        ----------
        frame_list : sequence[int]
            Frame indices in the batch.
        model : SurfaceModel, optional
            Pre-existing fitted model instance; a new
            :class:`HyperbolicTangentModel` is created if None.
        batch_index : int, optional
            Sequential identifier copied into the returned :class:`BinningBatch`
            (defaults to 1 when not supplied).

        Returns
        -------
        BinningBatch
            Per-batch container with contact angle, density field, fitted
            isoline coordinates and fitted parameters.
        """
        xi_par, zi_par, len_frames = self.get_profile_coordinates(
            frame_indices=frame_list,
        )
        n_particles = len(xi_par) / max(len_frames, 1)
        batch_label = f" {batch_index}" if batch_index is not None else ""
        logger.info(
            f"Number of fluid particles in batch{batch_label}: {n_particles:.2f}"
        )
        rho_cc = self.binning(xi_par, zi_par, len_frames)
        if model is None:
            model = HyperbolicTangentModel()
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(self.zi_cc, self.xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape(
            (len(self.xi_cc) * len(self.zi_cc)), order="F"
        )
        msh_xi_cc = msh_xi_cc_grid.reshape(
            (len(self.xi_cc) * len(self.zi_cc)), order="F"
        )
        msh_rho_cc = rho_cc.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        x_data = (msh_xi_cc, msh_zi_cc)
        model.fit(x_data, msh_rho_cc)
        logger.info(
            f"Fitted parameters for batch{batch_label}:\n"
            f"{''.join(model.get_parameter_strings())}"
        )
        contact_angle = model.compute_contact_angle()
        logger.info(f"Contact angle for batch{batch_label}: {contact_angle}")
        try:
            circle_xi, circle_zi, wall_line_xi, wall_line_zi = model.compute_isoline()
        except ValueError as exc:
            warnings.warn(
                f"Isoline unavailable for batch {batch_index}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            circle_xi = circle_zi = wall_line_xi = wall_line_zi = None
        params = model.params
        if params is None:
            raise RuntimeError(
                f"Hyperbolic tangent fit did not set model parameters for batch "
                f"{batch_index}; cannot build BinningBatch."
            )
        return BinningBatch(
            batch_index=batch_index if batch_index is not None else 1,
            angle=float(contact_angle),
            n_particles=float(n_particles),
            xi_cc=self.xi_cc.copy(),
            zi_cc=self.zi_cc.copy(),
            rho_cc=rho_cc,
            circle_xi=circle_xi,
            circle_zi=circle_zi,
            wall_line_xi=wall_line_xi,
            wall_line_zi=wall_line_zi,
            fitted_params=dict(zip(_PARAM_NAMES, params, strict=False)),
        )
