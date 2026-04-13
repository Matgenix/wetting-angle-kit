from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base_parser import BaseParser


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

    def load_xyz_file(self) -> List[Dict[str, Any]]:
        """Load all frames from the XYZ file into memory.

        Returns
        -------
        list[dict]
            Each entry has keys: ``symbols``, ``positions``, ``lattice_matrix``.
        """
        frames = []
        with open(self.filepath, "r") as file:
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
        self, liquid_particle_types: List[str], frame_index: int
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

    def get_profile_coordinates(
        self,
        frame_indices: Sequence[int],
        droplet_geometry: str = "cylinder_y",
        atom_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute 2D projection coordinates (r, z) for contact angle analysis.

        Projects 3D atomic positions onto a 2D plane based on the assumed
        droplet geometry and simulation box boundaries.

        Parameters
        ----------
        frame_indices : Sequence[int]
            List of frames to process.
        droplet_geometry : str, default 'cylinder_y'
            The physical shape of the water droplet in the simulation box:
            * 'cylinder_y': A hemi-cylindrical droplet aligned along the Y-axis.
               (Returns x as the radial coordinate).
            * 'cylinder_x': A hemi-cylindrical droplet aligned along the X-axis.
               (Returns y as the radial coordinate).
            * 'spherical': A spherical cap droplet.
               (Returns sqrt(x^2 + y^2) as the radial coordinate).
        atom_indices : Sequence[int], optional
            Subset of atom indices to include (e.g., only liquid atoms).

        Returns
        -------
        r_values : np.ndarray
            The lateral/radial distances from the droplet center/axis.
        z_values : np.ndarray
            The vertical coordinates (height) of the atoms.
        n_frames : int
            Number of frames processed.
        """
        r_values = np.array([])
        z_values = np.array([])
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            x_par = frame["positions"]
            if atom_indices is not None:
                atom_indices_arr = np.array(atom_indices)
                x_par = x_par[atom_indices_arr]
            x_cm = np.mean(x_par, axis=0)
            x_0 = x_par - x_cm
            x_0[:, 2] = x_par[:, 2]
            if droplet_geometry == "cylinder_y":
                r_frame = np.abs(x_0[:, 0] + 0.01)
            elif droplet_geometry == "cylinder_x":
                r_frame = np.abs(x_0[:, 1] + 0.01)
            else:  # spherical
                r_frame = np.sqrt(x_0[:, 0] ** 2 + x_0[:, 1] ** 2)
            z_frame = x_0[:, 2]
            r_values = np.concatenate((r_values, r_frame))
            z_values = np.concatenate((z_values, z_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {x_cm}")
        print(f"\nr range:\t({np.min(r_values)},{np.max(r_values)})")
        print(f"z range:\t({np.min(z_values)},{np.max(z_values)})")
        return r_values, z_values, len(frame_indices)

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

    def frame_count(self):
        """Return total number of frames loaded."""
        return len(self.frames)


class XYZWaterMoleculeFinder(BaseParser):
    """Parser specialized for identifying water oxygen atoms in XYZ trajectories.

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
        filepath,
        particle_type_wall,
        oxygen_type="O",
        hydrogen_type="H",
        oh_cutoff: float = 1.2,
    ):
        self.filepath = filepath
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load frames (without lattice) for water oxygen analysis."""
        frames = []
        with open(self.filepath, "r") as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            frame_start += 1  # skip comment
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {"symbols": np.array(symbols), "positions": np.array(positions)}
            )
            frame_start += num_atoms
        return frames

    def parse(self, liquid_particle_types: List[str], frame_index: int) -> np.ndarray:
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

    def get_profile_coordinates(
        self,
        frame_indices: Sequence[int],
        droplet_geometry: str = "cylinder_y",
        atom_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute 2D projection coordinates (r, z) for contact angle analysis.

        Projects 3D atomic positions onto a 2D plane based on the assumed
        droplet geometry and simulation box boundaries.

        Parameters
        ----------
        frame_indices : Sequence[int]
            List of frames to process.
        droplet_geometry : str, default 'cylinder_y'
            The physical shape of the water droplet in the simulation box:
            * 'cylinder_y': A hemi-cylindrical droplet aligned along the Y-axis.
               (Returns x as the radial coordinate).
            * 'cylinder_x': A hemi-cylindrical droplet aligned along the X-axis.
               (Returns y as the radial coordinate).
            * 'spherical': A spherical cap droplet.
               (Returns sqrt(x^2 + y^2) as the radial coordinate).
        atom_indices : Sequence[int], optional
            Subset of atom indices to include (e.g., only liquid atoms).

        Returns
        -------
        r_values : np.ndarray
            The lateral/radial distances from the droplet center/axis.
        z_values : np.ndarray
            The vertical coordinates (height) of the atoms.
        n_frames : int
            Number of frames processed.
        """
        r_values = np.array([])
        z_values = np.array([])
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            x_par = frame["positions"]
            if atom_indices is not None:
                atom_indices_arr = np.array(atom_indices)
                x_par = x_par[atom_indices_arr]
            x_cm = np.mean(x_par, axis=0)
            x_0 = x_par - x_cm
            x_0[:, 2] = x_par[:, 2]
            if droplet_geometry == "cylinder_y":
                r_frame = np.abs(x_0[:, 0] + 0.01)
            elif droplet_geometry == "cylinder_x":
                r_frame = np.abs(x_0[:, 1] + 0.01)
            else:
                r_frame = np.sqrt(x_0[:, 0] ** 2 + x_0[:, 1] ** 2)
            z_frame = x_0[:, 2]
            r_values = np.concatenate((r_values, r_frame))
            z_values = np.concatenate((z_values, z_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {x_cm}")
        print(f"\nr range:\t({np.min(r_values)},{np.max(r_values)})")
        print(f"z range:\t({np.min(z_values)},{np.max(z_values)})")
        return r_values, z_values, len(frame_indices)

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

    def get_water_oxygen_indices(self, frame_index):
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

    def get_water_oxygen_positions(self, frame_index):
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

    def _manual_water_identification(self, positions, oxygen_indices, hydrogen_indices):
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
