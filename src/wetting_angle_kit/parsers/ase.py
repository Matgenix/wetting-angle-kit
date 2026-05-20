from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from wetting_angle_kit.io_utils import assert_orthogonal_cell
from wetting_angle_kit.parsers.base import BaseParser


def _validate_ase_trajectory_orthogonal(trajectory: list) -> None:
    for idx, frame in enumerate(trajectory):
        assert_orthogonal_cell(np.asarray(frame.cell), context=f"Frame {idx}")


class AseParser(BaseParser):
    """ASE trajectory parser for any ASE-readable file format."""

    def __init__(self, filepath: str) -> None:
        """
        Parameters
        ----------
        filepath : str
            Path to any ASE-readable trajectory/file pattern (e.g. XYZ, extxyz,
            POSCAR, etc.).
        """
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'ase' package is required to use AseParser. Install with "
                "'pip install ase'."
            ) from e
        self.filepath = filepath
        self.trajectory = read(self.filepath, index=":")
        _validate_ase_trajectory_orthogonal(self.trajectory)

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Atom coordinates.
        """
        frame = self.trajectory[frame_index]
        if indices is not None:
            indices = np.array(indices)
            return frame.positions[indices]
        return frame.positions

    def parse_liquid_particles(
        self, liquid_particle_types: list[str], frame_index: int
    ) -> np.ndarray:
        """Return positions of liquid atoms filtered by atomic symbol.

        Parameters
        ----------
        liquid_particle_types : sequence[str]
            Symbols identifying liquid atoms.
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Liquid atom positions.
        """
        frame = self.trajectory[frame_index]
        mask = np.isin(frame.symbols, liquid_particle_types)
        return frame.positions[mask]

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell.lengths()[0])

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell.lengths()[1])

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
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return the total number of frames available."""
        return len(self.trajectory)


class AseWaterFinder:
    """Identify water oxygen atoms by counting hydrogen neighbors
    via ASE neighbor list."""

    def __init__(
        self,
        filepath: str,
        particle_type_wall: list[str],
        oxygen_type: str = "O",
        hydrogen_type: str = "H",
        oh_cutoff: float = 1.2,
    ):
        """
        Parameters
        ----------
        filepath : str
            Path to ASE-readable trajectory.
        particle_type_wall : sequence[str]
            Symbols representing wall particles (unused presently, reserved for
            filtering).
        oxygen_type : str, default "O"
            Oxygen atom symbol.
        hydrogen_type : str, default "H"
            Hydrogen atom symbol.
        oh_cutoff : float, default 1.2
            O–H distance cutoff used to detect bonded hydrogens.
        """
        try:
            from ase.io import read
            from ase.neighborlist import NeighborList
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use AseWaterFinder. "
                "Install it with: pip install ase"
            ) from e
        self._ase_read = read
        self._NeighborList = NeighborList
        self.trajectory = self._ase_read(filepath, index=":")
        _validate_ase_trajectory_orthogonal(self.trajectory)
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff

    def get_water_oxygen_indices(self, frame_index: int) -> np.ndarray:
        """Return indices of oxygen atoms bonded to exactly two hydrogens.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Oxygen atom indices belonging to water molecules.
        """
        frame = self.trajectory[frame_index]
        symbols = np.array(frame.get_chemical_symbols())
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_set = set(np.where(symbols == self.hydrogen_type)[0].tolist())
        # ASE's NeighborList uses pairwise cutoff = cutoffs[i] + cutoffs[j].
        # Use half the bond cutoff per atom so the effective pair cutoff
        # equals self.oh_cutoff.
        cutoffs = [self.oh_cutoff / 2.0] * len(frame)  # type: ignore[arg-type]
        nl = self._NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(frame)
        water_oxygens = []
        for o_idx in oxygen_indices:
            indices, _offsets = nl.get_neighbors(o_idx)
            h_count = sum(1 for i in indices if int(i) in hydrogen_set)
            if h_count == 2:
                water_oxygens.append(o_idx)
        return np.array(water_oxygens, dtype=int)

    def get_water_oxygen_positions(self, frame_index: int) -> np.ndarray:
        """Return positions of water oxygen atoms for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; empty array if none detected.
        """
        indices = self.get_water_oxygen_indices(frame_index)
        frame = self.trajectory[frame_index]
        return frame.positions[indices]


class AseWallParser(BaseParser):
    """Parser extracting wall particle coordinates (excluding liquid types).

    Wall particles are everything *not* in ``liquid_particle_types``. The
    ``indices`` argument of :meth:`parse` is treated as 0-based positional
    indices into the wall-only positions for compatibility with the
    :class:`BaseParser` contract.
    """

    def __init__(self, filepath: str, liquid_particle_types: list[str]):
        """
        Parameters
        ----------
        filepath : str
            Path to trajectory file.
        liquid_particle_types : sequence[str]
            Symbols representing liquid particles to exclude.
        """
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use AseWallParser. Install it "
                "with: pip install ase"
            ) from e
        self.filepath = filepath
        self.liquid_particle_types = liquid_particle_types
        self.trajectory = read(self.filepath, index=":")
        _validate_ase_trajectory_orthogonal(self.trajectory)

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return wall atom positions for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            Indices into the wall-only positions to further restrict the
            wall atoms; if None all wall atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Wall atoms coordinates.
        """
        frame = self.trajectory[frame_index]
        mask = ~np.isin(frame.get_chemical_symbols(), self.liquid_particle_types)
        x_par = frame.positions[mask]
        if indices is not None:
            x_par = x_par[np.asarray(indices, dtype=int)]
        return x_par

    def find_highest_wall_particle(self, frame_index: int) -> float:
        """Return the maximum z-coordinate among wall particles for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Maximum z-coordinate.
        """
        x_wall = self.parse(frame_index)
        return float(np.max(x_wall[:, 2]))

    def find_highest_wall_part(self, *args: Any, **kwargs: Any) -> float:
        """Deprecated alias for find_highest_wall_particle."""
        warnings.warn(
            "find_highest_wall_part is deprecated, "
            "use find_highest_wall_particle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_highest_wall_particle(*args, **kwargs)

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell.lengths()[0])

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell.lengths()[1])

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
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return the total number of frames available."""
        return len(self.trajectory)
