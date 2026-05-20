from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract interface for trajectory parsers consumed by analyzers.

    Subclasses must implement :meth:`parse` and :meth:`frame_count`. The
    geometric helpers (:meth:`box_size_x`, :meth:`box_size_y`,
    :meth:`box_length_max`) raise ``NotImplementedError`` by default and
    can be overridden where the underlying format exposes that information.
    """

    @abstractmethod
    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            Atom indices to select; if None all atoms are returned. The
            meaning of ``indices`` differs by parser because the underlying
            file formats do not all preserve atom ordering across frames:

            * :class:`LammpsDumpParser` (LAMMPS) — ``indices`` are LAMMPS
              particle identifiers. LAMMPS may reorder atoms between frames,
              so a persistent ID is needed to track the same atom.
            * :class:`XYZParser`, :class:`AseParser` — ``indices`` are
              positional indices into the per-frame coordinate array. These
              formats keep atom ordering stable across frames.

        Returns
        -------
        ndarray, shape (M, 3)
            Atom coordinates.
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
        """Return the x-dimension of the simulation box for a frame.

        Override in subclasses where the underlying format exposes it.
        """
        raise NotImplementedError("box_size_x not implemented for this parser.")

    def box_size_y(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the y-dimension of the simulation box for a frame.

        Override in subclasses where the underlying format exposes it.
        """
        raise NotImplementedError("box_size_y not implemented for this parser.")

    def box_length_max(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the maximum lattice vector length for a frame.

        Override in subclasses where the underlying format exposes it.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Max ``|a_i|`` over lattice vectors.
        """
        raise NotImplementedError("box_length_max not implemented for this parser.")
