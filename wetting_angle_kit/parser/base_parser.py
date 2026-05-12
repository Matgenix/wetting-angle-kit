from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np

from wetting_angle_kit.io_utils import validate_droplet_geometry

logger = logging.getLogger(__name__)


def project_to_profile(
    positions: np.ndarray, droplet_geometry: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D atomic positions onto the (r, z) plane used by analyzers.

    The lateral coordinates are centered on their per-frame center of mass
    before projection; the vertical (z) coordinate is left in lab frame.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian atomic positions for a single frame.
    droplet_geometry : str
        One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.

    Returns
    -------
    r_values : ndarray, shape (N,)
        Radial coordinate: |x_centered| for cylinder_y, |y_centered| for
        cylinder_x, sqrt(x_centered**2 + y_centered**2) for spherical.
    z_values : ndarray, shape (N,)
        Vertical coordinate (lab-frame z, not centered).
    """
    validate_droplet_geometry(droplet_geometry)
    if positions.size == 0:
        return np.empty(0), np.empty(0)
    x_cm = np.mean(positions, axis=0)
    x_centered = positions - x_cm
    # z stays in lab frame; analyzers need absolute heights to locate the wall.
    z_values = positions[:, 2]
    if droplet_geometry == "cylinder_y":
        r_values = np.abs(x_centered[:, 0])
    elif droplet_geometry == "cylinder_x":
        r_values = np.abs(x_centered[:, 1])
    else:  # spherical
        r_values = np.sqrt(x_centered[:, 0] ** 2 + x_centered[:, 1] ** 2)
    return r_values, z_values


class BaseParser(ABC):
    """Abstract interface for trajectory parsers consumed by analyzers.

    Subclasses must implement :meth:`parse` and :meth:`frame_count`. The
    geometric helpers (:meth:`box_size_x`, :meth:`box_size_y`,
    :meth:`box_length_max`) raise ``NotImplementedError`` by default and
    can be overridden where the underlying format exposes that information.
    The :meth:`get_profile_coordinates` projection is implemented once here
    using :meth:`parse` as the per-frame data hook.
    """

    @abstractmethod
    def parse(
        self, frame_index: int, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray[int], optional
            Atom selector; if None return all atoms. The meaning of
            ``indices`` differs by parser because the underlying file formats
            do not all preserve atom ordering across frames:

            * :class:`DumpParser` (LAMMPS) — ``indices`` are LAMMPS particle
              identifiers. LAMMPS may reorder atoms between frames, so a
              persistent ID is needed to track the same atom.
            * :class:`XYZParser`, :class:`AseParser` — ``indices`` are 0-based
              positional indices into the per-frame coordinate array. These
              formats keep atom ordering stable across frames.

        Returns
        -------
        ndarray, shape (M, 3)
            Particle coordinates.
        """

    @abstractmethod
    def frame_count(self) -> int:
        """Return the total number of frames available."""

    def frame_tot(self) -> int:
        """Return the total number of frames available. (Legacy name)."""
        import warnings

        warnings.warn(
            "frame_tot is deprecated, use frame_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frame_count()

    def box_size_x(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the box x-length for a frame. (override if available)."""
        raise NotImplementedError("box_size_x not implemented for this parser.")

    def box_size_y(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the box y-length for a frame. (override if available)."""
        raise NotImplementedError("box_size_y not implemented for this parser.")

    def box_length_max(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the maximum box length for a frame. (override if available)."""
        raise NotImplementedError("box_length_max not implemented for this parser.")

    def get_profile_coordinates(
        self,
        frame_indices: Sequence[int],
        droplet_geometry: str = "cylinder_y",
        atom_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute 2D projection coordinates (r, z) for contact angle analysis.

        Projects 3D atomic positions onto a 2D plane based on the assumed
        droplet geometry. Coordinates are accumulated across all requested
        frames in lockstep.

        Parameters
        ----------
        frame_indices : Sequence[int]
            Frame indices to process.
        droplet_geometry : str, default ``"cylinder_y"``
            ``"cylinder_y"`` returns ``|x|`` as the radial coordinate,
            ``"cylinder_x"`` returns ``|y|``, ``"spherical"`` returns
            ``sqrt(x^2 + y^2)``. ``x`` and ``y`` are centered on the
            per-frame center of mass; ``z`` is left in lab frame.
        atom_indices : Sequence[int], optional
            Atom selector forwarded to :meth:`parse` (see that method for
            the per-parser meaning of indices vs. identifiers).

        Returns
        -------
        r_values : ndarray
            Concatenated radial distances.
        z_values : ndarray
            Concatenated vertical coordinates.
        n_frames : int
            Number of frames processed (``len(frame_indices)``).
        """
        validate_droplet_geometry(droplet_geometry)
        r_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        for frame_idx in frame_indices:
            positions = self.parse(frame_idx, atom_indices)
            r_frame, z_frame = project_to_profile(positions, droplet_geometry)
            r_chunks.append(r_frame)
            z_chunks.append(z_frame)
            if frame_idx % 10 == 0:
                x_cm = (
                    np.mean(positions, axis=0) if positions.size else np.full(3, np.nan)
                )
                logger.info(
                    "Frame %d: %d particles, center of mass %s",
                    frame_idx,
                    len(positions),
                    np.array2string(x_cm, precision=3),
                )
        r_values = np.concatenate(r_chunks) if r_chunks else np.empty(0)
        z_values = np.concatenate(z_chunks) if z_chunks else np.empty(0)
        if r_values.size > 0:
            logger.info(
                "r range: (%.3f, %.3f)", float(r_values.min()), float(r_values.max())
            )
            logger.info(
                "z range: (%.3f, %.3f)", float(z_values.min()), float(z_values.max())
            )
        return r_values, z_values, len(frame_indices)
