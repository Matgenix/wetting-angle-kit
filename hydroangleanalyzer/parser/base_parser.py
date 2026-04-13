from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple

import numpy as np


class BaseParser(ABC):
    """Abstract interface for trajectory parsers consumed by analyzers.

    Subclasses must implement frame parsing and frame count methods. Optional
    geometry helpers (box size, cylindrical conversion) can be overridden where
    supported by underlying file format.

    Parameters
    ----------
    filepath : str
        Path to trajectory / structure file.
    particle_type_wall : Any
        Identifier(s) for wall particles (type IDs for LAMMPS dump, symbols for
        ASE/XYZ).
    """

    @abstractmethod
    def __init__(self, filepath: str, particle_type_wall: Any):
        pass

    @abstractmethod
    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray[int], optional
            Atom indices to select; if None return all atoms.

        Returns
        -------
        ndarray, shape (M, 3)
            Particle coordinates.
        """
        pass

    @abstractmethod
    def frame_count(self) -> int:
        """Return the total number of frames available.

        Returns
        -------
        int
            Number of frames.
        """
        pass

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
        raise NotImplementedError(
            "get_profile_coordinates not implemented for this parser."
        )
