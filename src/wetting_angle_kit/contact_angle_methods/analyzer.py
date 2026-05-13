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
    """Abstract base for contact angle analysis across trajectories."""

    @abstractmethod
    def analyze(
        self, frame_range: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the analysis and return statistics."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
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
    """
    This class is a wrapper around the ContactAngleSlicedParallel class.
    It is used to analyze the contact angle of a liquid on a solid surface.
    """

    def __init__(
        self,
        parser: Any,
        output_dir: str,
        **kwargs: Any,
    ):
        self.parser = parser
        self.output_dir = output_dir
        self._processor = ContactAngleSlicedParallel(
            filename=parser.filepath, output_dir=output_dir, **kwargs
        )

    def analyze(
        self, frame_range: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
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
        return "sliced_parallel"


class BinningContactAngleAnalyzer(BaseContactAngleAnalyzer):
    """
    This class is a wrapper around the ContactAngleBinning class.
    It is used to analyze the contact angle of a liquid on a solid surface.
    """

    def __init__(self, parser: Any, output_dir: str, **kwargs: Any) -> None:
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
        return "binning_density"
