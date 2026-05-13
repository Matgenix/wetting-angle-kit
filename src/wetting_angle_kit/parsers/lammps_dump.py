from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from wetting_angle_kit.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class LammpsDumpParser(BaseParser):
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
                "The 'ovito' package is required for LammpsDumpParser. Install with: "
                "pip install wetting_angle_kit[ovito]"
            ) from e

        # OVITO's type stubs return Optional[PipelineNode] and miss several
        # runtime attributes (SimulationCell.matrix, etc.). Cast to Any so
        # the type checker treats the pipeline as opaque.
        self.filepath = filepath
        self.pipeline: Any = cast(Any, import_file(self.filepath))
        self.pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        self.num_frames: int = int(self.pipeline.source.num_frames)

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
        return int(self.num_frames)


class LammpsDumpWallParser(BaseParser):
    """LAMMPS dump file parser for extracting wall particle coordinates.

    Wall particles are everything *not* in ``liquid_particle_types``;
    filtering is done inside the OVITO pipeline. The ``indices`` argument
    of :meth:`parse` is therefore typically ignored, but it is accepted
    (as LAMMPS particle IDs, like :class:`LammpsDumpParser`) to satisfy the
    :class:`BaseParser` contract.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    liquid_particle_types : List[int]
        Type IDs of particles to exclude as liquid.
    """

    def __init__(self, filepath: str, liquid_particle_types: list[int]):
        self.filepath = filepath
        self.liquid_particle_types = liquid_particle_types
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self) -> Any:
        """Build and return the OVITO pipeline for wall-only extraction.

        Returns ``Any`` because OVITO's Python bindings ship without type
        stubs; the pipeline is opaque from the type checker's perspective.
        """
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

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return wall particle positions for a single frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            LAMMPS particle IDs to further restrict the wall particles. If
            None, all wall particles are returned.
        """
        data = self.pipeline.compute(frame_index)
        x_par = np.asarray(data.particles["Position"])
        if indices is not None:
            particle_ids = np.asarray(data.particles["Particle Identifier"])
            mask = np.isin(particle_ids, indices)
            x_par = x_par[mask]
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

    def box_size_y(self, frame_index: int) -> float:
        """Return the y-dimension of the simulation box."""
        data = self.pipeline.compute(frame_index)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_size_x(self, frame_index: int) -> float:
        """Return the x-dimension of the simulation box."""
        data = self.pipeline.compute(frame_index)
        x_vector = data.cell.matrix[0, :3]
        return float(np.linalg.norm(x_vector))

    def box_length_max(self, frame_index: int) -> float:
        """Return the maximum simulation cell dimension for a frame."""
        data = self.pipeline.compute(frame_index)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return float(np.max(np.array([y_vector, x_vector, z_vector])))

    def frame_count(self) -> int:
        """Return total number of frames."""
        return int(self.pipeline.source.num_frames)


class LammpsDumpWaterFinder:
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

    def _setup_pipeline(self) -> Any:
        """Setup OVITO pipeline for water molecule detection.

        Returns ``Any`` because OVITO's stubs disagree with the runtime
        API; treat the pipeline as opaque from the type checker's
        perspective.
        """
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
        pipeline: Any = cast(Any, import_file(self.filepath))
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
        if "IsWaterOxygen" not in data.particles:
            raise RuntimeError(
                "OVITO pipeline did not produce the 'IsWaterOxygen' property. "
                "Check that the CoordinationAnalysisModifier and "
                "ComputePropertyModifier ran successfully and that the "
                "oxygen_type/hydrogen_type values match the trajectory."
            )
        mask = np.array(data.particles["IsWaterOxygen"].array) == 1
        oxygen_indices = np.where(mask)[0]
        return np.asarray(data.particles["Particle Identifier"][oxygen_indices])
