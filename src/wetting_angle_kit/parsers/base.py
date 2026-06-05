from abc import ABC, abstractmethod

import numpy as np


class BaseParser(ABC):
    """Abstract interface for trajectory parsers consumed by analyzers.

    Subclasses must implement :meth:`parse`, :meth:`frame_count`, and the
    cell-geometry helpers :meth:`box_size_x`, :meth:`box_size_y`, and
    :meth:`box_length_max`. The cell helpers are abstract because the
    analyzers rely on per-frame box information (PBC-aware droplet
    recentering, default sampling extent); a parser without it would
    silently degrade their accuracy.
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
              formats keep atom ordering stable across frames during simulations.

        Returns
        -------
        ndarray, shape (M, 3)
            Atom coordinates.
        """

    @abstractmethod
    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory."""

    @abstractmethod
    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""

    @abstractmethod
    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""

    @abstractmethod
    def box_length_max(self, frame_index: int) -> float:
        """Return the maximum lattice vector length for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Max ``|a_i|`` over lattice vectors.
        """
