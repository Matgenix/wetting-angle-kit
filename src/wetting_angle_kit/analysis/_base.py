"""Shared worker-pool scaffolding for batched trajectory analyzers.

:class:`_BatchedTrajectoryAnalyzer` is the private base from which both
:class:`TrajectoryAnalyzer` and the two coupled-fit analyzers
(:class:`CoupledBinning2DAnalyzer`, :class:`CoupledBinning3DAnalyzer`)
inherit. It centralises:

- the common constructor (parser + atom_indices + droplet_geometry +
  temporal_aggregator + precentered + optional wall_atom_indices);
- the :meth:`analyze` orchestration (spawn-context worker pool + tqdm
  progress + out-of-order result reassembly);
- the per-batch coordinate gathering (per-frame PBC recentering and
  the ``cylinder_x`` axis swap, pooled across the batch's frames);
- helpers for building a parser inside a worker.

Subclasses fill in:

- ``_init_args()`` / ``_init_worker(...)`` — what shared state to
  send to worker processes;
- ``_process_batch_worker(frame_indices)`` — the per-batch
  computation, run inside a worker;
- ``_build_results(batches)`` — packaging the per-batch results
  into the analyzer's results dataclass.

Algorithm bodies for the subclass extension points are stubbed at this
skeleton stage. The shared scaffolding (constructor, ``analyze``, coord
gathering) is real because it is small, well-defined, and stable.
"""

import logging
import multiprocessing as mp
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from tqdm.auto import tqdm

from wetting_angle_kit.analysis.analyzer import BaseTrajectoryAnalyzer
from wetting_angle_kit.analysis.geometry import DropletGeometry
from wetting_angle_kit.analysis.temporal import TemporalAggregator
from wetting_angle_kit.io_utils import (
    detect_parser_type,
    recenter_droplet_pbc,
)
from wetting_angle_kit.parsers.ase import AseParser
from wetting_angle_kit.parsers.base import BaseParser
from wetting_angle_kit.parsers.lammps_dump import LammpsDumpParser
from wetting_angle_kit.parsers.xyz import XYZParser

# Spawn is required because parser instances may hold un-picklable
# handles (OVITO pipelines, ASE Atoms with C extensions). A scoped
# context keeps this side-effect-free at import time.
_MP_CONTEXT = mp.get_context("spawn")

logger = logging.getLogger(__name__)


def build_parser(filename: str) -> BaseParser:
    """Build a parser by sniffing the file's extension.

    Used by worker processes to rebuild a parser locally from a
    filepath, since the parent's parser instance is generally not
    picklable.
    """
    parser_type = detect_parser_type(filename)
    if parser_type == "dump":
        return LammpsDumpParser(filepath=filename)
    if parser_type == "ase":
        return AseParser(filepath=filename)
    if parser_type == "xyz":
        return XYZParser(filepath=filename)
    raise ValueError(f"Unsupported parser type: {parser_type}")


def gather_batch_coords(
    *,
    parser: BaseParser,
    frame_indices: list[int],
    atom_indices: np.ndarray,
    droplet_geometry: DropletGeometry,
    precentered: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pool liquid-atom coordinates across one batch.

    Each frame's atoms are recentered (per-frame circular-mean PBC
    recentering unless ``precentered=True``) and the ``cylinder_x``
    axis swap is applied so downstream code can assume the cylinder
    axis is always ``y``. Per-frame coordinate arrays are then
    concatenated.

    Parameters
    ----------
    parser : BaseParser
        Parser to read frames from. Typically the worker-local copy.
    frame_indices : list[int]
        Frames to pool. May be a single frame (per-frame analysis) or
        many frames (batched / fully pooled).
    atom_indices : ndarray
        Indices of liquid atoms; passed to ``parser.parse``.
    droplet_geometry : DropletGeometry
        Used for both PBC recentering and the axis swap.
    precentered : bool
        If True, skip the circular-mean PBC recentering and use the
        plain arithmetic-mean centre.

    Returns
    -------
    pooled_coords : ndarray, shape (sum_N, 3)
        Concatenated liquid coordinates in the internal frame.
    avg_center : ndarray, shape (3,)
        Mean of the per-frame liquid centres; used as the ray-fan
        origin by extractors.
    max_dist : float
        Maximum in-plane box half-extent across the batch, used as
        the radial-sampling envelope.
    """
    liquid_chunks: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    max_box = 0.0
    for frame_num in frame_indices:
        positions = parser.parse(frame_index=frame_num, indices=atom_indices)
        if precentered:
            mean_pos = np.mean(positions, axis=0)
        else:
            box_xy = (
                parser.box_size_x(frame_index=frame_num),
                parser.box_size_y(frame_index=frame_num),
            )
            positions, mean_pos = recenter_droplet_pbc(
                positions, droplet_geometry.name, box_size=box_xy
            )
        # The axis swap on a (N, 3) array; DropletGeometry handles
        # both batch and single-point cases via the trailing-axis
        # fancy index.
        positions = droplet_geometry.to_internal_coords(positions)
        mean_pos = droplet_geometry.to_internal_coords(mean_pos)
        liquid_chunks.append(positions)
        centres.append(mean_pos)
        box_x = parser.box_size_x(frame_index=frame_num)
        box_y = parser.box_size_y(frame_index=frame_num)
        max_box = max(max_box, float(box_x), float(box_y))
    pooled = (
        np.concatenate(liquid_chunks, axis=0) if liquid_chunks else np.empty((0, 3))
    )
    avg_center = np.mean(np.stack(centres, axis=0), axis=0) if centres else np.zeros(3)
    return pooled, avg_center, max_box / 2.0


def gather_wall_coords(
    *,
    parser: BaseParser,
    frame_indices: list[int],
    wall_atom_indices: np.ndarray,
    droplet_geometry: DropletGeometry,
) -> np.ndarray:
    """Pool wall-atom coordinates across one batch.

    Wall atoms are not PBC-recentered (the wall is assumed fixed in
    its lab-frame position) but the ``cylinder_x`` axis swap is
    applied so the wall lives in the same internal frame as the
    liquid pool returned by :func:`gather_batch_coords`.

    Parameters
    ----------
    parser : BaseParser
        Parser to read frames from.
    frame_indices : list[int]
        Frames to pool.
    wall_atom_indices : ndarray
        Indices of wall atoms.
    droplet_geometry : DropletGeometry
        Used for the axis swap.

    Returns
    -------
    ndarray, shape (sum_N, 3)
        Concatenated wall coordinates in the internal frame.
    """
    chunks: list[np.ndarray] = []
    for frame_num in frame_indices:
        positions = parser.parse(frame_index=frame_num, indices=wall_atom_indices)
        positions = droplet_geometry.to_internal_coords(positions)
        chunks.append(positions)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))


class _BatchedTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """Shared scaffolding for batched trajectory analyzers.

    Not user-facing. Concrete analyzers (:class:`TrajectoryAnalyzer`,
    :class:`CoupledBinning2DAnalyzer`, :class:`CoupledBinning3DAnalyzer`)
    inherit from this and provide the four extension points listed in
    the module docstring.
    """

    #: Per-process worker state. Each concrete subclass MUST shadow this
    #: with its own empty dict so worker initialisation writes to a
    #: subclass-specific slot rather than colliding via the parent class.
    _WORKER_STATE: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        parser: Any,
        atom_indices: np.ndarray | None = None,
        droplet_geometry: DropletGeometry | str = "spherical",
        *,
        temporal_aggregator: TemporalAggregator | None = None,
        precentered: bool = False,
        wall_atom_indices: np.ndarray | None = None,
    ) -> None:
        # Fail fast on the parser shape so users see the error at
        # construction, not on first batch.
        detect_parser_type(parser.filepath)
        self.parser = parser
        self.atom_indices = (
            np.asarray(atom_indices) if atom_indices is not None else np.array([])
        )
        self.wall_atom_indices = (
            np.asarray(wall_atom_indices) if wall_atom_indices is not None else None
        )
        self.droplet_geometry = DropletGeometry.coerce(droplet_geometry)
        self.temporal_aggregator = temporal_aggregator or TemporalAggregator(
            batch_size=1
        )
        self.precentered = precentered

    def analyze(
        self,
        frame_range: list[int] | None = None,
        n_jobs: int | None = 1,
    ) -> Any:
        """Run the analyzer across batches.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to analyse. Defaults to every frame in the
            trajectory (``range(parser.frame_count())``).
        n_jobs : int, default 1
            Worker process count. ``1`` (default) runs in-process
            with no ``multiprocessing.Pool`` overhead. Set to an
            integer ``>= 2`` for parallel execution, or to ``None``
            to let ``multiprocessing.Pool`` pick the default
            (``os.cpu_count()``).

        Returns
        -------
        Results
            Subclass-specific results dataclass produced by
            :meth:`_build_results`.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        batches = list(self.temporal_aggregator.iter_batches(frame_range))
        if not batches:
            return self._build_results(batches=[])

        logger.info(
            f"Processing {len(batches)} batches "
            f"(batch_size={self.temporal_aggregator.batch_size}, n_jobs={n_jobs})."
        )

        init_args = self._init_args()
        if n_jobs == 1 or len(batches) == 1:
            batch_results = self._run_inline(batches, init_args)
        else:
            batch_results = self._run_parallel(batches, init_args, n_jobs)

        if not batch_results:
            raise RuntimeError(
                f"None of the {len(batches)} requested batches produced a "
                "result. Check the worker logs above for the underlying "
                "parser, geometry, or fit errors."
            )

        # ``imap_unordered`` returns completion-ordered; restore batch
        # order using the first frame index in each batch.
        batch_results.sort(key=lambda b: min(b.frames) if b.frames else 0)
        return self._build_results(batches=batch_results)

    def _run_inline(
        self,
        batches: list[list[int]],
        init_args: tuple,
    ) -> list[Any]:
        """In-process batch loop; avoids ``Pool`` spawn overhead."""
        self._init_worker(*init_args)
        results: list[Any] = []
        with tqdm(
            total=len(batches),
            desc=self._tqdm_desc(),
            unit="batch",
        ) as pbar:
            for batch in batches:
                result = self._process_batch_worker(batch)
                if result is not None:
                    results.append(result)
                pbar.update(1)
        return results

    def _run_parallel(
        self,
        batches: list[list[int]],
        init_args: tuple,
        n_jobs: int | None,
    ) -> list[Any]:
        """Multi-process batch loop via ``multiprocessing.Pool``."""
        results: list[Any] = []
        with (
            _MP_CONTEXT.Pool(
                processes=n_jobs,
                initializer=self._init_worker,
                initargs=init_args,
            ) as pool,
            tqdm(
                total=len(batches),
                desc=self._tqdm_desc(),
                unit="batch",
            ) as pbar,
        ):
            for result in pool.imap_unordered(self._process_batch_worker, batches):
                if result is not None:
                    results.append(result)
                pbar.update(1)
        return results

    def _tqdm_desc(self) -> str:
        """Progress bar label. Subclasses may override for clarity."""
        return type(self).__name__

    # ------------------------------------------------------------------
    # Subclass extension points.
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_args(self) -> tuple:
        """Return the tuple of args sent to every worker on startup.

        Must be picklable. The companion :meth:`_init_worker` unpacks
        this tuple inside each worker and stores the rebuilt state.
        """

    @staticmethod
    @abstractmethod
    def _init_worker(*args: Any) -> None:
        """Populate the subclass's ``_WORKER_STATE`` inside a worker.

        Called once per worker process at pool startup. Implementations
        typically rebuild the parser via :func:`build_parser` and
        stash all per-batch-needed state into the class-level dict.
        """

    @staticmethod
    @abstractmethod
    def _process_batch_worker(frame_indices: list[int]) -> Any:
        """Process one batch inside a worker process.

        Reads from the subclass's ``_WORKER_STATE`` populated by
        :meth:`_init_worker`. Returns the per-batch result (a
        :class:`BatchResult` subclass for :class:`TrajectoryAnalyzer`,
        a :class:`CoupledBinning2DBatchResult` for the 2D coupled fit,
        etc.) — or ``None`` on a per-batch failure. The parent process
        logs and skips ``None`` results, raising only if every batch
        fails.
        """

    @abstractmethod
    def _build_results(self, batches: list[Any]) -> Any:
        """Wrap per-batch results into the analyzer's results dataclass.

        ``batches`` carries the concrete per-batch type the subclass
        returns from :meth:`_process_batch_worker`.
        """
