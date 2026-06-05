from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wetting_angle_kit.io_utils import assert_orthogonal_cell
from wetting_angle_kit.parsers.base import BaseParser


class XYZParser(BaseParser):
    """Extended XYZ trajectory parser."""

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Path to extended XYZ trajectory containing per-frame atom count,
            comment line with lattice vectors, then atom symbol + coordinates.
        """
        self.filepath = filepath
        self.frames = self.load_xyz_file()

    def load_xyz_file(self) -> list[dict[str, Any]]:
        """Load all frames from the XYZ file into memory using numpy.

        Returns
        -------
        list[dict]
            Each entry has keys: ``symbols``, ``positions``, ``lattice_matrix``.
        """
        frames: list[dict[str, np.ndarray]] = []
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
            assert_orthogonal_cell(lattice_matrix, context=f"Frame {len(frames)}")
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
        indices : ndarray, optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Atom coordinates.
        """
        frame = self.frames[frame_index]
        if indices is not None and len(indices) > 0:
            indices = np.array(indices, dtype=int)
            return frame["positions"][indices]
        return frame["positions"]

    def parse_liquid_particles(
        self, liquid_particle_types: list[str], frame_index: int
    ) -> np.ndarray:
        """Return positions of liquid atoms identified by
        their atomic symbol.

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
        frame = self.frames[frame_index]
        mask = np.isin(frame["symbols"], liquid_particle_types)
        return frame["positions"][mask]

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.linalg.norm(lattice_matrix[0]))

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.linalg.norm(lattice_matrix[1]))

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

    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory."""
        return len(self.frames)


class XYZWaterFinder:
    """Helper for identifying water oxygen atoms in XYZ trajectories.

    This is a standalone helper (not a :class:`BaseParser`) because its
    ``parse`` signature filters by atomic symbol rather than frame index,
    which is incompatible with the parser ABC contract.
    """

    def __init__(
        self,
        filepath: str,
        oxygen_type: str = "O",
        hydrogen_type: str = "H",
        oh_cutoff: float = 1.2,
    ):
        """
        Parameters
        ----------
        filepath : str
            Path to XYZ file.
        oxygen_type : str, default "O"
            Oxygen atom symbol.
        hydrogen_type : str, default "H"
            Hydrogen atom symbol.
        oh_cutoff : float, default 1.2
            Distance cutoff (Å) for O-H bonding to identify water molecules.
        """
        self.filepath = filepath
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.frames = self.load_xyz_file()

    def load_xyz_file(self) -> list[dict[str, Any]]:
        """Load frames including the lattice matrix for box-size queries."""
        frames: list[dict[str, np.ndarray | None]] = []
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
                assert_orthogonal_cell(lattice_matrix, context=f"Frame {len(frames)}")
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
        data = self.frames[frame_index]
        positions = data["positions"]
        symbols = data["symbols"]
        lattice_matrix = data.get("lattice_matrix")
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        return self._manual_water_identification(
            positions, oxygen_indices, hydrogen_indices, lattice_matrix
        )

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
        lattice_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Identify water oxygens by counting hydrogens within cutoff distance.

        Uses a :class:`scipy.spatial.cKDTree` over the hydrogen positions.
        When ``lattice_matrix`` is provided, the kd-tree's ``boxsize`` is
        set from its diagonal (the cell is enforced orthogonal upstream) so
        O–H pairs are matched under minimum-image convention; otherwise the
        match is done in open space.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Atomic positions.
        oxygen_indices : ndarray
            Candidate oxygen indices.
        hydrogen_indices : ndarray
            Hydrogen indices to check.
        lattice_matrix : ndarray, shape (3, 3), optional
            Orthogonal cell. If given, pairwise distances are evaluated
            under PBC; otherwise raw Cartesian distances are used.

        Returns
        -------
        ndarray
            Oxygen indices with exactly two nearby hydrogens.
        """
        if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
            return np.array([], dtype=int)

        o_pos = positions[oxygen_indices]
        h_pos = positions[hydrogen_indices]

        if lattice_matrix is not None:
            # Orthogonal cell — diagonal entries are the axis-aligned box
            # lengths. cKDTree requires coordinates inside ``[0, L)``, so
            # wrap with a modulo before building the tree.
            box = np.abs(np.diag(np.asarray(lattice_matrix, dtype=float)))
            o_pos = o_pos - np.floor(o_pos / box) * box
            h_pos = h_pos - np.floor(h_pos / box) * box
            tree = cKDTree(h_pos, boxsize=box)
        else:
            tree = cKDTree(h_pos)

        neighbours = tree.query_ball_point(o_pos, r=self.oh_cutoff)
        h_counts = np.fromiter(
            (len(n) for n in neighbours), dtype=int, count=len(o_pos)
        )
        return oxygen_indices[h_counts == 2]


class XYZWallParser(BaseParser):
    """Parser extracting wall particle coordinates from an XYZ trajectory.

    Wall particles are everything *not* in ``liquid_particle_types``; the
    mask is applied at :meth:`parse` time over the per-frame symbol array.
    The ``indices`` argument of :meth:`parse` is treated as 0-based
    positional indices into the wall-only positions, mirroring
    :class:`~wetting_angle_kit.parsers.ase.AseWallParser`.
    """

    def __init__(self, filepath: str, liquid_particle_types: list[str]) -> None:
        """
        Parameters
        ----------
        filepath : str
            Path to extended XYZ trajectory.
        liquid_particle_types : sequence[str]
            Atomic symbols representing liquid particles to exclude.
        """
        self.filepath = filepath
        self.liquid_particle_types = liquid_particle_types
        # Reuse ``XYZParser`` for loading: it already validates orthogonal
        # cells and stores symbols + positions + lattice per frame.
        self.frames = XYZParser(filepath).frames

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return wall atom positions for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            0-based indices into the wall-only positions to further
            restrict the result; if None all wall atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Wall atom coordinates.
        """
        frame = self.frames[frame_index]
        mask = ~np.isin(frame["symbols"], self.liquid_particle_types)
        x_par = frame["positions"][mask]
        if indices is not None:
            x_par = x_par[np.asarray(indices, dtype=int)]
        return x_par

    def find_highest_wall_particle(self, frame_index: int) -> float:
        """Return the maximum z-coordinate among wall particles for a frame."""
        x_wall = self.parse(frame_index)
        return float(np.max(x_wall[:, 2]))

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.linalg.norm(lattice_matrix[0]))

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.linalg.norm(lattice_matrix[1]))

    def box_length_max(self, frame_index: int) -> float:
        """Return the maximum lattice vector length for a frame."""
        lattice_matrix = self.frames[frame_index]["lattice_matrix"]
        return float(np.max(np.linalg.norm(lattice_matrix, axis=1)))

    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory."""
        return len(self.frames)
