import os
import shutil
import tempfile

import pytest


@pytest.fixture
def trajectory_file_path():
    """Return the path to the test trajectory file."""
    return "trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_trajectory(trajectory_file_path, temp_output_dir):
    """
    Create a small sample trajectory file for testing.
    Skips if the source file does not exist.
    """
    if not os.path.exists(trajectory_file_path):
        pytest.skip(f"Source trajectory file '{trajectory_file_path}' not found")

    sample_path = os.path.join(temp_output_dir, "sample_trajectory.lammpstrj")

    try:
        with open(trajectory_file_path, "r") as source, open(sample_path, "w") as dest:
            for _ in range(1000):  # Copy first 1000 lines
                line = source.readline()
                if not line:
                    break
                dest.write(line)
    except Exception as e:
        pytest.skip(f"Failed to create sample trajectory: {e}")

    yield sample_path
