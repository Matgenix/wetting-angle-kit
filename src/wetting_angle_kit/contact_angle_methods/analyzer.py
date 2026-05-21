from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from wetting_angle_kit.contact_angle_methods.binning.angle_fitting import (
    ContactAngleBinning,
)
from wetting_angle_kit.contact_angle_methods.sliced.parallel import (
    ContactAngleSlicedParallel,
)


class BaseContactAngleAnalyzer(ABC):
    """Abstract base for contact angle analysis across trajectory files."""

    @abstractmethod
    def analyze(
        self, frame_range: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the analysis and return statistics."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the method name identifier."""
        pass

    def summary(self) -> dict[str, float]:
        """Return quick summary statistics."""
        results = self.analyze()
        return {
            "mean": results["mean_angle"],
            "std": results["std_angle"],
            "n_samples": len(results["angles"]),
        }


class SlicedContactAngleAnalyzer(BaseContactAngleAnalyzer):
    """BaseContactAngleAnalyzer implementation using the sliced parallel method."""

    def __init__(
        self,
        parser: Any,
        output_dir: str,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser instance.
        output_dir : str
            Directory for output files.
        **kwargs
            Forwarded to ContactAngleSlicedParallel.
        """
        self.parser = parser
        self.output_dir = output_dir
        self._processor = ContactAngleSlicedParallel(
            filename=parser.filepath, output_dir=output_dir, **kwargs
        )

    def analyze(
        self, frame_range: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the sliced parallel analysis and return statistics.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. If None, all frames are used.
        **kwargs
            Forwarded to process_frames_parallel.

        Returns
        -------
        dict
            Keys: mean_angle, std_angle, angles, frames_analyzed, method_metadata.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))

        frame_to_angle = self._processor.process_frames_parallel(
            frames_to_process=frame_range, **kwargs
        )
        angles = np.array(list(frame_to_angle.values()))

        return {
            "mean_angle": np.mean(angles),
            "std_angle": np.std(angles),
            "angles": frame_to_angle,
            "frames_analyzed": list(frame_to_angle.keys()),
            "method_metadata": {"frames_per_angle": 1},
        }

    def get_method_name(self) -> str:
        """Return "sliced_parallel"."""
        return "sliced_parallel"


class BinningContactAngleAnalyzer(BaseContactAngleAnalyzer):
    """BaseContactAngleAnalyzer implementation using the density-binning method."""

    def __init__(self, parser: Any, output_dir: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser instance.
        output_dir : str
            Directory for output files.
        **kwargs
            Forwarded to ContactAngleBinning.
        """
        self.parser = parser
        self.output_dir = output_dir
        self._analyzer = ContactAngleBinning(
            parser=parser, output_dir=output_dir, **kwargs
        )

    def analyze(
        self,
        frame_range: list[int] | None = None,
        split_factor: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the binning analysis and return statistics.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. If None, all frames are used.
        split_factor : int, optional
            If given, split frame_range into sub-batches of this size and
            compute one angle per batch; if None, all frames form a single batch.
        **kwargs
            Reserved for future use.

        Returns
        -------
        dict
            Keys: mean_angle, std_angle, angles, frames_analyzed, method_metadata.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        if split_factor is None:
            angle, _ = self._analyzer.process_batch(frame_range)
            angles = np.array([angle])
            method_metadata = {"frames_per_angle": len(frame_range)}
        else:
            angles_list: list[float] = []
            for batch_idx, start in enumerate(range(0, len(frame_range), split_factor)):
                end = min(start + split_factor, len(frame_range))
                angle, _ = self._analyzer.process_batch(
                    frame_range[start:end],
                    batch_index=batch_idx + 1,  # Pass batch index
                )
                angles_list.append(angle)
            angles = np.array(angles_list)
            method_metadata = {"frames_per_trajectory": split_factor}
        return {
            "mean_angle": np.mean(angles),
            "std_angle": np.std(angles),
            "angles": angles,
            "frames_analyzed": frame_range,
            "method_metadata": method_metadata,
        }

    def get_method_name(self) -> str:
        """Return "binning_density"."""
        return "binning_density"
