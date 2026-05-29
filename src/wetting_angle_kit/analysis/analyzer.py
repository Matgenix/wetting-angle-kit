from abc import ABC, abstractmethod
from typing import Any

from wetting_angle_kit.analysis.binning.angle_fitting import (
    ContactAngleBinning,
)
from wetting_angle_kit.analysis.binning.results import BinningResults
from wetting_angle_kit.analysis.slicing.parallel import (
    ContactAngleSlicingParallel,
)
from wetting_angle_kit.analysis.slicing.results import SlicingResults


class BaseContactAngleAnalyzer(ABC):
    """Abstract base for contact angle analysis across trajectory files."""

    @abstractmethod
    def analyze(self, frame_range: list[int] | None = None, **kwargs: Any) -> Any:
        """Run the analysis and return a method-specific results object."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the method name identifier."""
        pass


class SlicingContactAngleAnalyzer(BaseContactAngleAnalyzer):
    """BaseContactAngleAnalyzer implementation using the slicing parallel method."""

    def __init__(
        self,
        parser: Any,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser instance.
        **kwargs
            Forwarded to ContactAngleSlicingParallel.
        """
        self.parser = parser
        self._processor = ContactAngleSlicingParallel(
            filename=parser.filepath, **kwargs
        )

    def analyze(
        self, frame_range: list[int] | None = None, **kwargs: Any
    ) -> SlicingResults:
        """Run the slicing parallel analysis.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. If None, all frames are used.
        **kwargs
            Forwarded to ``process_frames_parallel``.

        Returns
        -------
        SlicingResults
            In-memory per-frame angles, surface contours and circle fits.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        return self._processor.process_frames_parallel(
            frames_to_process=frame_range, **kwargs
        )

    def get_method_name(self) -> str:
        """Return "slicing_parallel"."""
        return "slicing_parallel"


class BinningContactAngleAnalyzer(BaseContactAngleAnalyzer):
    """BaseContactAngleAnalyzer implementation using the density-binning method."""

    def __init__(self, parser: Any, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser instance.
        **kwargs
            Forwarded to ContactAngleBinning.
        """
        self.parser = parser
        self._analyzer = ContactAngleBinning(parser=parser, **kwargs)

    def analyze(
        self,
        frame_range: list[int] | None = None,
        split_factor: int | None = None,
        **kwargs: Any,
    ) -> BinningResults:
        """Run the binning analysis.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. If None, all frames are used.
        split_factor : int, optional
            If given, split ``frame_range`` into sub-batches of this size and
            compute one angle per batch; if None, all frames form a single batch.
        **kwargs
            Reserved for future use.

        Returns
        -------
        BinningResults
            Per-batch contact angles, density fields and isoline data.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        if split_factor is None:
            batch = self._analyzer.process_batch(frame_range)
            return BinningResults(
                batches=[batch],
                method_metadata={"frames_per_angle": len(frame_range)},
            )
        batches = []
        for batch_idx, start in enumerate(range(0, len(frame_range), split_factor)):
            end = min(start + split_factor, len(frame_range))
            batches.append(
                self._analyzer.process_batch(
                    frame_range[start:end],
                    batch_index=batch_idx + 1,
                )
            )
        return BinningResults(
            batches=batches,
            method_metadata={"frames_per_trajectory": split_factor},
        )

    def get_method_name(self) -> str:
        """Return "binning_density"."""
        return "binning_density"
