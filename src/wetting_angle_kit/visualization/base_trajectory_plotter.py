from abc import ABC, abstractmethod

from wetting_angle_kit.visualization.stats import TrajectoryStats


class BaseTrajectoryPlotter(ABC):
    """Abstract base for trajectory plotters.

    Subclasses own their own data layout (per-method result containers,
    directories, etc.) and must implement :meth:`summary` returning one
    :class:`TrajectoryStats` per trajectory.
    """

    @abstractmethod
    def summary(self) -> list[TrajectoryStats]:
        """Return per-trajectory summary statistics."""
