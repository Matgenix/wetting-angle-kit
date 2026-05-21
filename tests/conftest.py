"""Shared pytest fixtures and constants for the test suite.

Note: most tests build their own trajectory paths via
``os.path.join(os.path.dirname(__file__), "../trajectories/...")``. The
fixtures here are reserved for cases that need a tmp copy of a fixture
trajectory or a shared output directory.
"""

import os

#: Absolute path to the directory containing fixture trajectories.
TRAJECTORIES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "trajectories")
)


def trajectory_path(name: str) -> str:
    """Return absolute path to a fixture trajectory by file name."""
    return os.path.join(TRAJECTORIES_DIR, name)
