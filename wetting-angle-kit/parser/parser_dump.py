from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class DumpParser(BaseParser):
    """LAMMPS dump trajectory parser.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    """

    def __init__(self, filepath: str):
        """Initialize LAMMPS dump parser via OVITO pipeline."""
        try:
            from ovito.io import import_file
            from ovito.modifiers import ComputePropertyModifier
        except ImportError as e:
            raise ImportError(
                "The 'ovito' package is required for DumpParser. Install with: "
                "pip install wetting_angle_kit[ovito]"
            ) from e

        self.filepath = filepath
        self.pipeline = import_file(self.filepath)
        self.pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        self.num_frames = self.pipeline.source.num_frames

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Compute and return particle positions for a single frame,
        with optional filtering by particle indices.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            Atom indices to select; if None return all atoms.

        Returns
        -------
        ndarray, shape (M, 3)
            Particle coordinates.
        """
        data = self.pipeline.compute(frame_index)
        x_par = np.asarray(data.particles["Position"])
        particle_ids = np.asarray(data.particles["Particle Identifier"])
        if indices is not None:
            mask = np.isin(particle_ids, indices)
            x_par = x_par[mask]
        return x_par

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
            x_par = self.parse(frame_idx, atom_indices)
            # dim = x_par.shape[1]
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

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension of simulation box."""
        data = self.pipeline.compute(frame_index)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_size_x(self, frame_index: int) -> float:
        """Return x-dimension of simulation box."""
        data = self.pipeline.compute(frame_index)
        x_vector = data.cell.matrix[0, :3]
        return float(np.linalg.norm(x_vector))

    def box_length_max(self, frame_index: int) -> float:
        """Return the maximum dimension of the simulation box.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Maximum box length.
        """
        data = self.pipeline.compute(frame_index)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return float(np.max(np.array([y_vector, x_vector, z_vector])))

    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory.

        Returns
        -------
        int
            Number of frames.
        """
        return self.num_frames


class DumpWallParser:
    """LAMMPS dump file parser for extracting wall particle coordinates.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    liquid_particle_types : List[int]
        Type IDs of particles to exclude as liquid.
    """

    def __init__(self, filepath: str, liquid_particle_types: List[int]):
        self.filepath = filepath
        self.liquid_particle_types = liquid_particle_types
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self):
        try:
            from ovito.io import import_file
            from ovito.modifiers import (
                ComputePropertyModifier,
                DeleteSelectedModifier,
                SelectTypeModifier,
            )
        except ImportError as e:
            raise ImportError(
                "OVITO required. Install with: pip install wetting_angle_kit[ovito]"
            ) from e
        pipeline = import_file(self.filepath)
        pipeline.modifiers.append(
            SelectTypeModifier(
                property="Particle Type", types=self.liquid_particle_types
            )
        )
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        return pipeline

    def parse(self, frame_index):
        """Compute and return wall particle positions for a single frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        np.ndarray
            Array of wall particle positions.
        """
        data = self.pipeline.compute(frame_index)
        return np.asarray(data.particles["Position"])

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
        data = self.pipeline.compute(frame_index)
        x_wall = np.asarray(data.particles["Position"])
        return float(np.max(x_wall[:, 2]))

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
            data = self.pipeline.compute(frame_idx)
            x_par = np.asarray(data.particles["Position"])
            if atom_indices is not None:
                # In DumpWallParser, we use Particle Identifier to filter
                # if indices are provided
                # But DumpWallParser seems to be designed to
                # exclude liquid types already.
                # However, for consistency with the interface:
                particle_ids = np.asarray(data.particles["Particle Identifier"])
                mask = np.isin(particle_ids, atom_indices)
                x_par = x_par[mask]

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

    def box_size_y(self, frame_index: int) -> float:
        """Return the y-dimension of the simulation box."""
        data = self.pipeline.compute(frame_index)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_length_max(self, frame_index):  # legacy name kept
        data = self.pipeline.compute(frame_index)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector, x_vector, z_vector]))

    def frame_count(self) -> int:
        """Return total number of frames."""
        return self.pipeline.source.num_frames


class DumpWaterMoleculeFinder:
    """Identify water oxygen atoms in a parsed LAMMPS trajectory."""

    def __init__(
        self,
        filepath: str,
        particle_type_wall: set,
        oxygen_type: int = 3,
        hydrogen_type: int = 2,
        oh_cutoff: float = 1.2,
    ):
        """Initialize water molecule finder with OVITO pipeline."""
        self.filepath = filepath
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.pipeline = self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup OVITO pipeline for water molecule detection."""
        try:
            from ovito.io import import_file
            from ovito.modifiers import (
                ComputePropertyModifier,
                CoordinationAnalysisModifier,
            )
        except ImportError as e:
            raise ImportError(
                "OVITO required for water detection. Install: pip install "
                "wetting_angle_kit[ovito]"
            ) from e
        pipeline = import_file(self.filepath)
        pipeline.modifiers.append(
            CoordinationAnalysisModifier(cutoff=self.oh_cutoff, number_of_bins=200)
        )
        expr = f"(ParticleType == {self.oxygen_type}) && (Coordination == 2)"
        pipeline.modifiers.append(
            ComputePropertyModifier(expressions=[expr], output_property="IsWaterOxygen")
        )
        return pipeline

    def get_water_oxygen_ids(self, frame_index: int) -> np.ndarray:
        """Return IDs of oxygen atoms belonging to water molecules."""
        data = self.pipeline.compute(frame_index)
        if "IsWaterOxygen" in data.particles:
            mask = np.array(data.particles["IsWaterOxygen"].array) == 1
            oxygen_indices = np.where(mask)[0]
            oxygen_ids = data.particles["Particle Identifier"][oxygen_indices]
            return oxygen_ids
        return self._manual_identification(data)


Dump_WaterMoleculeFinder = DumpWaterMoleculeFinder
DumpParse_wall = DumpWallParser
