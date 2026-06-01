"""Trajectory-level slicing contact-angle analyzer."""

import logging
import multiprocessing as mp
from typing import Any, NamedTuple

import numpy as np

from wetting_angle_kit.analysis.analyzer import BaseTrajectoryAnalyzer
from wetting_angle_kit.analysis.slicing.angle_fitting import (
    SlicingFrameFitter,
)
from wetting_angle_kit.analysis.slicing.results import SlicingResults
from wetting_angle_kit.io_utils import (
    detect_parser_type,
    recenter_droplet_pbc,
    validate_droplet_geometry,
)
from wetting_angle_kit.parsers.ase import AseParser
from wetting_angle_kit.parsers.base import BaseParser
from wetting_angle_kit.parsers.lammps_dump import LammpsDumpParser
from wetting_angle_kit.parsers.xyz import XYZParser

# "spawn" is required because parser instances may hold un-picklable handles
# (OVITO pipelines, ASE Atoms with C extensions). Using a scoped context
# rather than mutating the global start method keeps this side-effect-free
# when the package is imported.
_MP_CONTEXT = mp.get_context("spawn")

logger = logging.getLogger(__name__)


class _SlicingFrameResult(NamedTuple):
    """Per-frame output of the slicing worker."""

    frame_num: int
    mean_angle: float | None
    angles: list
    surfaces: list
    popts: list


class SlicingTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """Trajectory-level slicing contact-angle analyzer.

    Frames are dispatched one-by-one to a ``multiprocessing.Pool`` whose
    workers each build their own parser once and reuse it for every frame
    they receive. The per-frame fitting work is delegated to
    :class:`SlicingFrameFitter`.
    """

    # Per-worker state populated by ``_init_worker`` in each child process.
    # In the parent this stays empty; ``spawn`` gives each child its own
    # fresh module-level class object, so the dict is effectively per-process.
    _WORKER_STATE: dict[str, Any] = {}

    def __init__(
        self,
        parser: Any,
        droplet_geometry: str = "spherical",
        atom_indices: np.ndarray | None = None,
        delta_gamma: float | None = None,
        delta_cylinder: float | None = None,
        points_per_angstrom: float = 1.0,
        precentered: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser instance. Only ``parser.filepath`` and
            ``parser.frame_count()`` are read in the parent process; each
            worker rebuilds its own parser from ``filepath``.
        droplet_geometry : str, default "spherical"
            One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.
        atom_indices : ndarray, optional
            Indices of liquid particles. Empty array selects none.
        delta_gamma : float, optional
            Azimuthal step (degrees) for spherical analysis (required if
            ``droplet_geometry == "spherical"``).
        delta_cylinder : float, optional
            Slice spacing along the cylinder axis (required for
            cylinder_x / cylinder_y).
        points_per_angstrom : float, default 1.0
            Sampling density along each radial ray.
        precentered : bool, default False
            Skip per-frame circular-mean PBC recentering. Setting this on a
            trajectory that does NOT satisfy the precondition will produce
            wrong results.
        """
        # Fail fast in the parent process so the user gets the error at
        # construction instead of a uniform "all frames failed" later.
        detect_parser_type(parser.filepath)
        validate_droplet_geometry(droplet_geometry)
        if droplet_geometry == "spherical":
            if delta_gamma is None:
                raise ValueError("delta_gamma must be provided for spherical analysis")
            if delta_cylinder is not None:
                raise ValueError(
                    "delta_cylinder must not be set for spherical analysis "
                    "(it is only valid for cylinder_x / cylinder_y)."
                )
        elif droplet_geometry in ("cylinder_x", "cylinder_y"):
            if delta_cylinder is None:
                raise ValueError(
                    f"delta_cylinder must be provided for {droplet_geometry}."
                )
            if delta_gamma is not None:
                raise ValueError(
                    f"delta_gamma must not be set for {droplet_geometry} "
                    "(it is only valid for spherical)."
                )
        self.parser = parser
        self.droplet_geometry = droplet_geometry
        self.atom_indices = atom_indices if atom_indices is not None else np.array([])
        self.delta_gamma = delta_gamma
        self.delta_cylinder = delta_cylinder
        self.points_per_angstrom = points_per_angstrom
        self.precentered = precentered

    def analyze(
        self,
        frame_range: list[int] | None = None,
        n_jobs: int | None = None,
    ) -> SlicingResults:
        """Run the slicing analysis in parallel across frames.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. Defaults to all frames.
        n_jobs : int, optional
            Number of worker processes. ``None`` lets ``multiprocessing.Pool``
            pick the default (``os.cpu_count()``).

        Returns
        -------
        SlicingResults
            Per-frame angles, surface contours, fit parameters and method
            metadata. Frames whose worker failed to produce a mean angle are
            omitted.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        if not frame_range:
            return SlicingResults(
                frames=[],
                angles=[],
                surfaces=[],
                popts=[],
                method_metadata={"frames_per_angle": 1},
            )
        init_args = (
            self.parser.filepath,
            self.droplet_geometry,
            self.atom_indices,
            self.delta_gamma,
            self.delta_cylinder,
            self.points_per_angstrom,
            self.precentered,
        )
        logger.info(f"Processing {len(frame_range)} frames with n_jobs={n_jobs}")
        results_by_frame: dict[int, _SlicingFrameResult] = {}
        with _MP_CONTEXT.Pool(
            processes=n_jobs,
            initializer=self._init_worker,
            initargs=init_args,
        ) as pool:
            for result in pool.imap_unordered(self._run_one_frame, frame_range):
                if result.mean_angle is not None:
                    results_by_frame[result.frame_num] = result
        sorted_frames = sorted(results_by_frame)
        logger.info(
            f"Successfully processed {len(sorted_frames)}/{len(frame_range)} frames"
        )
        if not sorted_frames:
            raise RuntimeError(
                f"None of the {len(frame_range)} requested frames produced "
                "any contact-angle slices. Check the worker logs above for the "
                "underlying parser, geometry, or fit errors."
            )
        return SlicingResults(
            frames=sorted_frames,
            angles=[np.asarray(results_by_frame[f].angles) for f in sorted_frames],
            surfaces=[results_by_frame[f].surfaces for f in sorted_frames],
            popts=[np.asarray(results_by_frame[f].popts) for f in sorted_frames],
            method_metadata={"frames_per_angle": 1},
        )

    @staticmethod
    def _build_parser(filename: str) -> BaseParser:
        parser_type = detect_parser_type(filename)
        if parser_type == "dump":
            return LammpsDumpParser(filepath=filename)
        if parser_type == "ase":
            return AseParser(filepath=filename)
        if parser_type == "xyz":
            return XYZParser(filepath=filename)
        raise ValueError(f"Unsupported parser type: {parser_type}")

    @staticmethod
    def _init_worker(
        filename: str,
        droplet_geometry: str,
        atom_indices: np.ndarray,
        delta_gamma: float | None,
        delta_cylinder: float | None,
        points_per_angstrom: float,
        precentered: bool,
    ) -> None:
        cls = SlicingTrajectoryAnalyzer
        cls._WORKER_STATE.clear()
        cls._WORKER_STATE.update(
            parser=cls._build_parser(filename),
            droplet_geometry=droplet_geometry,
            atom_indices=atom_indices,
            delta_gamma=delta_gamma,
            delta_cylinder=delta_cylinder,
            points_per_angstrom=points_per_angstrom,
            precentered=precentered,
        )

    @staticmethod
    def _run_one_frame(frame_num: int) -> _SlicingFrameResult:
        state = SlicingTrajectoryAnalyzer._WORKER_STATE
        parser: BaseParser = state["parser"]
        droplet_geometry: str = state["droplet_geometry"]
        atom_indices: np.ndarray = state["atom_indices"]
        delta_gamma = state["delta_gamma"]
        delta_cylinder = state["delta_cylinder"]
        points_per_angstrom: float = state["points_per_angstrom"]
        precentered: bool = state["precentered"]
        try:
            liquid_positions = parser.parse(
                frame_index=frame_num,
                indices=atom_indices,
            )
            max_dist = int(
                np.max(
                    np.array(
                        [
                            parser.box_size_y(frame_index=frame_num),
                            parser.box_size_x(frame_index=frame_num),
                        ]
                    )
                )
                / 2
            )
            # Fold the droplet into the minimum-image frame around its
            # circular-mean COM before any cylinder_x axis swap, so the
            # ``box_size`` argument is in the parser's native frame. This
            # makes downstream radial sampling robust to droplets that
            # straddle a periodic boundary, and is idempotent for
            # trajectories already recentered during dynamics. Skipped
            # (with a plain arithmetic mean) when the user has declared
            # the trajectory pre-centered.
            if precentered:
                mean_liquid_position = np.mean(liquid_positions, axis=0)
            else:
                box_size_xy = (
                    parser.box_size_x(frame_index=frame_num),
                    parser.box_size_y(frame_index=frame_num),
                )
                liquid_positions, mean_liquid_position = recenter_droplet_pbc(
                    liquid_positions, droplet_geometry, box_size=box_size_xy
                )
            if droplet_geometry == "cylinder_x":
                liquid_positions = liquid_positions[:, [1, 0, 2]]
                mean_liquid_position = mean_liquid_position[[1, 0, 2]]
                box_dimensions = parser.box_size_x(frame_index=frame_num)
            elif droplet_geometry == "cylinder_y":
                box_dimensions = parser.box_size_y(frame_index=frame_num)
            else:
                box_dimensions = None
            predictor = SlicingFrameFitter(
                liquid_coordinates=liquid_positions,
                max_dist=max_dist,
                liquid_geom_center=mean_liquid_position,
                droplet_geometry=droplet_geometry,
                delta_gamma=delta_gamma,
                width_cylinder=box_dimensions,
                delta_cylinder=delta_cylinder,
                points_per_angstrom=points_per_angstrom,
            )
            angles, surfaces, popt_arrays = predictor.predict_contact_angle()
            if not angles:
                logger.warning(f"Frame {frame_num}: No angles computed (empty list).")
                return _SlicingFrameResult(frame_num, None, [], [], [])
            mean_angle = float(np.mean(angles))
            logger.info(f"Frame {frame_num} - mean angle: {mean_angle:.2f}°")
            return _SlicingFrameResult(
                frame_num, mean_angle, angles, surfaces, popt_arrays
            )
        except Exception as e:
            logger.error(f"Error processing frame {frame_num}: {e}", exc_info=True)
            return _SlicingFrameResult(frame_num, None, [], [], [])
