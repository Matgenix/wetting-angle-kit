from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base_parser import BaseParser


class AseParser(BaseParser):
    """ASE trajectory parser.

    Parameters
    ----------
    filepath : str
        Path to any ASE-readable trajectory/file pattern (e.g. XYZ, extxyz,
        POSCAR, etc.).
    """

    def __init__(self, filepath: str) -> None:
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'ase' package is required to use AseParser. Install with "
                "'pip install ase'."
            ) from e
        self.filepath = filepath
        self.trajectory = read(self.filepath, index=":")

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame from there index
        or the all frame if no index is provided.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : sequence[int], optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Cartesian coordinates of requested atoms.
        """
        frame = self.trajectory[frame_index]
        if indices is not None:
            indices = np.array(indices)
            return frame.positions[indices]
        return frame.positions

    def parse_liquid_particles(
        self, liquid_particle_types: List[str], frame_index: int
    ) -> np.ndarray:
        """Return liquid atom coordinates filtering by atomic symbol list.

        Parameters
        ----------
        liquid_particle_types : sequence[str]
            Symbols identifying liquid particles.
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
            frame = self.trajectory[frame_idx]
            x_par = frame.positions
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

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[1, 1])

    def box_size_x(self, frame_index: int) -> float:
        """Return x-dimension (a1x) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[0, 0])

    def box_length_max(self, frame_index: int) -> float:
        """Return maximum lattice vector length for frame."""
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return total number of frames in trajectory."""
        return len(self.trajectory)


class AseWaterMoleculeFinder:
    """Identify water oxygen atoms by counting hydrogen neighbors.

    Uses ASE neighbor list to find oxygens with exactly two hydrogens.

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

    def __init__(
        self,
        filepath: str,
        particle_type_wall: List[str],
        oxygen_type: str = "O",
        hydrogen_type: str = "H",
        oh_cutoff: float = 1.2,
    ):
        try:
            from ase.io import read
            from ase.neighborlist import NeighborList
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use AseWaterMoleculeFinder. "
                "Install it with: pip install ase"
            ) from e
        self._ase_read = read
        self._NeighborList = NeighborList
        self.trajectory = self._ase_read(filepath, index=":")
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff

    def get_water_oxygen_indices(self, frame_index: int) -> np.ndarray:
        """Return indices of oxygen atoms each bonded to exactly two hydrogens.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Oxygen atom indices satisfying bonding criterion.
        """
        frame = self.trajectory[frame_index]
        symbols = np.array(frame.get_chemical_symbols())
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        cutoffs = [self.oh_cutoff] * len(frame)
        nl = self._NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(frame)
        water_oxygens = []
        for o_idx in oxygen_indices:
            indices, _offsets = nl.get_neighbors(o_idx)
            h_neighbors = [i for i in indices if i in hydrogen_indices]
            if len(h_neighbors) == 2:
                water_oxygens.append(o_idx)
        return np.array(water_oxygens, dtype=int)

    def get_water_oxygen_positions(self, frame_index: int) -> np.ndarray:
        """Return Cartesian positions of water oxygen atoms for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; may be empty if none match criteria.
        """
        indices = self.get_water_oxygen_indices(frame_index)
        frame = self.trajectory[frame_index]
        return frame.positions[indices]


class AseWallParser:
    """Parser extracting wall particle coordinates (excluding liquid types).

    Parameters
    ----------
    filepath : str
        Path to trajectory file.
    liquid_particle_types : sequence[str]
        Symbols representing liquid particles to exclude.
    """

    def __init__(self, filepath: str, liquid_particle_types: List[str]):
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

    def parse(self, frame_index: int) -> np.ndarray:
        """Return wall coordinates for the supplied frame index.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Wall particle coordinates.
        """
        frame = self.trajectory[frame_index]
        mask = ~np.isin(frame.get_chemical_symbols(), self.liquid_particle_types)
        return frame.positions[mask]

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

    def find_highest_wall_part(self, *args, **kwargs):
        """Deprecated alias for find_highest_wall_particle."""
        warnings.warn(
            "find_highest_wall_part is deprecated, "
            "use find_highest_wall_particle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_highest_wall_particle(*args, **kwargs)

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
            frame = self.trajectory[frame_idx]
            x_par = frame.positions
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

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[1, 1])

    def box_length_max(self, frame_index: int) -> float:
        """Return maximum lattice vector length for frame."""
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return total number of frames in trajectory."""
        return len(self.trajectory)

    def frame_tot(self) -> int:
        """Deprecated alias for frame_count."""
        warnings.warn(
            "frame_tot is deprecated, use frame_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frame_count()
