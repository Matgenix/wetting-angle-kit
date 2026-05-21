from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from wetting_angle_kit.io_utils import assert_orthogonal_cell, ovito_cell_vectors
from wetting_angle_kit.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class LammpsDumpParser(BaseParser):
    """LAMMPS dump trajectory parser backed by an OVITO pipeline."""

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Path to LAMMPS dump file.
        """
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
        self._orthogonal_validated: set[int] = set()

    def _compute(self, frame_index: int) -> Any:
        """Compute a frame and validate its cell is orthogonal (once per frame)."""
        data = self.pipeline.compute(frame_index)
        if frame_index not in self._orthogonal_validated:
            assert_orthogonal_cell(
                ovito_cell_vectors(data), context=f"Frame {frame_index}"
            )
            self._orthogonal_validated.add(frame_index)
        return data

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            LAMMPS particle IDs to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Atom coordinates.
        """
        data = self._compute(frame_index)
        x_par = np.asarray(data.particles["Position"])
        particle_ids = np.asarray(data.particles["Particle Identifier"])
        if indices is not None:
            mask = np.isin(particle_ids, indices)
            x_par = x_par[mask]
        return x_par

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        data = self._compute(frame_index)
        return float(np.linalg.norm(ovito_cell_vectors(data)[:, 0]))

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        data = self._compute(frame_index)
        return float(np.linalg.norm(ovito_cell_vectors(data)[:, 1]))

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
        data = self._compute(frame_index)
        return float(np.max(np.linalg.norm(ovito_cell_vectors(data), axis=0)))

    def frame_count(self) -> int:
        """Return the total number of frames available."""
        return int(self.num_frames)


class LammpsDumpWallParser(BaseParser):
    """LAMMPS dump file parser for extracting wall particle coordinates.

    Wall particles are everything *not* in ``liquid_particle_types``;
    filtering is done inside the OVITO pipeline. The ``indices`` argument
    of :meth:`parse` is therefore typically ignored, but it is accepted
    (as LAMMPS particle IDs, like :class:`LammpsDumpParser`) to satisfy the
    :class:`BaseParser` contract.
    """

    def __init__(self, filepath: str, liquid_particle_types: list[int]):
        """
        Parameters
        ----------
        filepath : str
            Path to LAMMPS dump file.
        liquid_particle_types : List[int]
            Type IDs of particles to exclude as liquid.
        """
        self.filepath = filepath
        self.liquid_particle_types = liquid_particle_types
        self.pipeline = self.load_dump_ovito()
        self._orthogonal_validated: set[int] = set()

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

    def _compute(self, frame_index: int) -> Any:
        """Compute a frame and validate its cell is orthogonal (once per frame)."""
        data = self.pipeline.compute(frame_index)
        if frame_index not in self._orthogonal_validated:
            assert_orthogonal_cell(
                ovito_cell_vectors(data), context=f"Frame {frame_index}"
            )
            self._orthogonal_validated.add(frame_index)
        return data

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return wall atom positions for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            LAMMPS particle IDs to further restrict the wall atoms; if
            None all wall atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Wall atom coordinates.
        """
        data = self._compute(frame_index)
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

    def box_size_x(self, frame_index: int) -> float:
        """Return the length of the first lattice vector for a frame."""
        data = self._compute(frame_index)
        return float(np.linalg.norm(ovito_cell_vectors(data)[:, 0]))

    def box_size_y(self, frame_index: int) -> float:
        """Return the length of the second lattice vector for a frame."""
        data = self._compute(frame_index)
        return float(np.linalg.norm(ovito_cell_vectors(data)[:, 1]))

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
        data = self._compute(frame_index)
        return float(np.max(np.linalg.norm(ovito_cell_vectors(data), axis=0)))

    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory."""
        return int(self.pipeline.source.num_frames)


class LammpsDumpWaterFinder:
    """Identify water oxygen atoms in a LAMMPS trajectory via an OVITO pipeline."""

    def __init__(
        self,
        filepath: str,
        particle_type_wall: set,
        oxygen_type: int = 3,
        hydrogen_type: int = 2,
        oh_cutoff: float = 1.2,
    ):
        """
        Parameters
        ----------
        filepath : str
            Path to LAMMPS dump file.
        particle_type_wall : set
            Particle type IDs corresponding to wall atoms (reserved for future
            filtering).
        oxygen_type : int, default 3
            LAMMPS particle type ID for oxygen atoms.
        hydrogen_type : int, default 2
            LAMMPS particle type ID for hydrogen atoms.
        oh_cutoff : float, default 1.2
            O-H distance cutoff (Å) for water molecule detection.
        """
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
        """Return LAMMPS particle IDs of oxygen atoms bonded to exactly two hydrogens.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Oxygen particle IDs belonging to water molecules.
        """
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
