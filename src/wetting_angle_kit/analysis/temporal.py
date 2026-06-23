"""Temporal aggregation across frames for trajectory analysis.

A :class:`TemporalAggregator` groups frame indices into batches. Each
batch is later processed by an analyzer as a single fitting unit,
producing one contact angle estimate per batch.

The ``batch_size`` parameter controls the time-vs-statistics trade-off:

- ``batch_size=1`` (default) — per-frame analysis. Produces a time
  series with one angle per frame; statistics come from frame-to-frame
  variation.
- ``batch_size=N`` (N > 1) — pool consecutive groups of ``N`` frames
  together before fitting. Reduces thermal noise per fit at the cost
  of time resolution.
- ``batch_size=-1`` — fully pooled. Every requested frame goes into a
  single batch, producing one angle estimate for the trajectory.
"""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalAggregator:
    """Group frame indices into batches for per-batch surface fitting.

    Designed to be held by a :class:`TrajectoryAnalyzer` and driven from
    inside :meth:`analyze`, which supplies the frame indices to walk.
    Standalone use is fine for inspection (e.g. previewing batch
    boundaries) but the caller must always provide ``frame_range``.

    Parameters
    ----------
    batch_size : int, default 1
        Number of consecutive frames pooled per surface fit.
        ``batch_size=1`` (the default) gives per-frame analysis: each
        frame is its own batch. Larger values pool consecutive groups
        of frames, trading time resolution for statistics; the last
        batch is shorter if the range isn't evenly divisible.
        ``batch_size=-1`` is the "all" sentinel: every supplied frame
        is pooled into a single batch.
    """

    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size == 0 or self.batch_size < -1:
            raise ValueError(
                f"batch_size must be a positive integer or -1 (pool all); "
                f"got {self.batch_size!r}."
            )

    def iter_batches(self, frame_range: list[int]) -> Iterator[list[int]]:
        """Yield successive lists of frame indices, one per fitting unit.

        Parameters
        ----------
        frame_range : list[int]
            The frame indices to distribute. The analyzer normally
            populates this with ``range(parser.frame_count())`` or with
            a caller-supplied subset; the aggregator only groups what
            it is given and never consults the parser itself. May be
            empty (no batches yielded).

        Yields
        ------
        list[int]
            One batch of frame indices. Order within and across batches
            preserves the order of ``frame_range``.
        """
        if not frame_range:
            return
        if self.batch_size == -1:
            yield list(frame_range)
            return
        for i in range(0, len(frame_range), self.batch_size):
            yield list(frame_range[i : i + self.batch_size])

    def n_batches(self, n_frames: int) -> int:
        """Return the number of batches that would be yielded.

        Useful for sizing progress bars before iteration starts.

        Parameters
        ----------
        n_frames : int
            Length of the ``frame_range`` that would be passed to
            :meth:`iter_batches`.

        Returns
        -------
        int
            Number of batches the aggregator will produce for that input.
        """
        if n_frames <= 0:
            return 0
        if self.batch_size == -1:
            return 1
        return -(-n_frames // self.batch_size)
