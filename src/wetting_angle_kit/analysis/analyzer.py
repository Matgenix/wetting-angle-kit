"""Abstract base class for contact-angle analyzers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTrajectoryAnalyzer(ABC):
    """Abstract base for contact angle analysis across trajectory files."""

    @abstractmethod
    def analyze(self, frame_range: list[int] | None = None) -> Any:
        """Run the analysis and return a method-specific results object."""
        pass
