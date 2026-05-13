from __future__ import annotations

from typing import Any

import numpy as np

from wetting_angle_kit.parser.base_parser import BaseParser


class XYZParser(BaseParser):
    """XYZ trajectory parser.

    Parameters
    ----------
    filepath : str
        Path to extended XYZ trajectory containing per-frame atom count, comment line
        with lattice vectors, then atom symbol + coordinates.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.frames = self.load_xyz_file()

    def load_xyz_file(self) -> list[dict[str, Any]]:
        """Load all frames from the XYZ file into memory.

        Returns
        -------
        list[dict]
            Each entry has keys: ``symbols``, ``positions``, ``lattice_matrix``.
        """
        frames = []
        with open(self.filepath) as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            comment_line = lines[frame_start].strip()
            lattice_info = comment_line.split('Lattice="')[1].split('"')[0]
            lattice_vectors = np.array(lattice_info.split(), dtype=float)
            lattice_matrix = lattice_vectors.reshape(3, 3)
            frame_start += 1
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {
                    "symbols": np.array(symbols),
                    "positions": np.array(positions),
                    "lattice_matrix": lattice_matrix,
                }
            )
            frame_start += num_atoms
        return frames

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : sequence[int], optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Coordinates of requested atoms.
        """
        frame = self.frames[frame_index]
        if indices is not None and len(indices) > 0:
            indices = np.array(indices, dtype=int)
            return frame["positions"][indices]
        return frame["positions"]

    def parse_liquid_particles(
        self, liquid_particle_types: list[str], frame_index: int
    ) -> np.ndarray:
        """Return positions of liquid particles (filter by symbols).

        Parameters
        ----------
        liquid_particle_types : sequence[str]
            Atom symbols considered liquid (e.g. water molecule constituents).
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Cartesian coordinates of liquid atoms.
        """
        frame = self.frames[frame_index]
        mask = np.isin(frame["symbols"], liquid_particle_types)
        return frame["positions"][mask]

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
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.max(np.linalg.norm(lattice_matrix, axis=1)))

    def box_size_x(self, frame_index: int) -> float:
        """Return the box length along x (a1x component).

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Box x-length.
        """
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(lattice_matrix[0, 0])

    def box_size_y(self, frame_index: int) -> float:
        """Return the box length along y (a2y component).

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Box y-length.
        """
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(lattice_matrix[1, 1])

    def frame_count(self) -> int:
        """Return total number of frames loaded."""
        return len(self.frames)


class XYZWaterMoleculeFinder:
    """Helper for identifying water oxygen atoms in XYZ trajectories.

    This is a standalone helper (not a :class:`BaseParser`) because its
    ``parse`` signature filters by atomic symbol rather than frame index,
    which is incompatible with the parser ABC contract.

    Parameters
    ----------
    filepath : str
        Path to XYZ file.
    particle_type_wall : sequence[str]
        Symbols that represent wall (excluded) particles.
    oxygen_type : str, default "O"
        Oxygen atom symbol.
    hydrogen_type : str, default "H"
        Hydrogen atom symbol.
    oh_cutoff : float, default 1.2
        Distance cutoff (Å) for O-H bonding to identify water molecules.
    """

    def __init__(
        self,
        filepath: str,
        particle_type_wall: Any,
        oxygen_type: str = "O",
        hydrogen_type: str = "H",
        oh_cutoff: float = 1.2,
    ):
        self.filepath = filepath
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.frames = self.load_xyz_file()

    def load_xyz_file(self) -> list[dict[str, Any]]:
        """Load frames including the lattice matrix for box-size queries."""
        frames = []
        with open(self.filepath) as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            comment_line = lines[frame_start].strip()
            lattice_matrix: np.ndarray | None
            if 'Lattice="' in comment_line:
                lattice_info = comment_line.split('Lattice="')[1].split('"')[0]
                lattice_vectors = np.array(lattice_info.split(), dtype=float)
                lattice_matrix = lattice_vectors.reshape(3, 3)
            else:
                lattice_matrix = None
            frame_start += 1
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {
                    "symbols": np.array(symbols),
                    "positions": np.array(positions),
                    "lattice_matrix": lattice_matrix,
                }
            )
            frame_start += num_atoms
        return frames

    def parse(self, liquid_particle_types: list[str], frame_index: int) -> np.ndarray:
        """Return liquid particle coordinates filtering wall types.

        Parameters
        ----------
        liquid_particle_types : sequence[str]
            Symbols for liquid particles.
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Liquid atom positions.
        """
        frame = self.frames[frame_index]
        mask = np.isin(frame["symbols"], liquid_particle_types)
        return frame["positions"][mask]

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
        lattice_matrix = self.frames[frame_index].get("lattice_matrix")
        if lattice_matrix is None:
            raise ValueError(
                "No Lattice= entry in the XYZ comment line for frame "
                f"{frame_index}; box length cannot be determined."
            )
        return float(np.max(np.linalg.norm(lattice_matrix, axis=1)))

    def get_water_oxygen_indices(self, frame_index: int) -> np.ndarray:
        """Return indices of oxygen atoms belonging to water molecules.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Indices of oxygen atoms with exactly two hydrogens within cutoff.
        """
        data = self.frames[frame_index]
        positions = data["positions"]
        symbols = data["symbols"]
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        return self._manual_water_identification(
            positions, oxygen_indices, hydrogen_indices
        )

    def get_water_oxygen_positions(self, frame_index: int) -> np.ndarray:
        """Return coordinates of water oxygen atoms for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; empty array if none detected.
        """
        positions = self.frames[frame_index]["positions"]
        indices = self.get_water_oxygen_indices(frame_index)
        if len(indices) == 0:
            return np.empty((0, 3))
        return positions[indices]

    def _manual_water_identification(
        self,
        positions: np.ndarray,
        oxygen_indices: np.ndarray,
        hydrogen_indices: np.ndarray,
    ) -> np.ndarray:
        """Identify water oxygens by counting hydrogens within cutoff distance.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Atomic positions.
        oxygen_indices : ndarray
            Candidate oxygen indices.
        hydrogen_indices : ndarray
            Hydrogen indices to check.

        Returns
        -------
        ndarray
            Oxygen indices with exactly two nearby hydrogens.
        """
        water_oxygens = []
        for o_idx in oxygen_indices:
            o_pos = positions[o_idx]
            h_count = 0
            for h_idx in hydrogen_indices:
                h_pos = positions[h_idx]
                if np.linalg.norm(o_pos - h_pos) <= self.oh_cutoff:
                    h_count += 1
            if h_count == 2:
                water_oxygens.append(o_idx)
        return np.array(water_oxygens)
