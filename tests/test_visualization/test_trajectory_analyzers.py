import os

import numpy as np
import pytest

from wetting_angle_kit.visualization.binning_trajectory_analyzer import (
    BinningTrajectoryAnalyzer,
)
from wetting_angle_kit.visualization.slicing_trajectory_analyzer import (
    SlicingTrajectoryAnalyzer,
)


def _square_polygon(side: float = 2.0) -> np.ndarray:
    half = side / 2.0
    return np.array(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ]
    )


@pytest.fixture
def slicing_result_dir(tmp_path):
    """Create a directory with the .npy files expected by SlicingTrajectoryAnalyzer."""
    directory = tmp_path / "slicing_result"
    directory.mkdir()
    polygon = _square_polygon(side=4.0)
    all_angles = np.array(
        [
            (0, np.array([85.0, 90.0, 95.0])),
            (1, np.array([87.0, 92.0, 96.0])),
        ],
        dtype=object,
    )
    all_surfaces = np.array(
        [
            (0, [polygon, polygon * 1.1]),
            (1, [polygon * 1.05, polygon * 1.15]),
        ],
        dtype=object,
    )
    all_popts = np.array(
        [
            (0, np.array([1.0, 2.0, 3.0, 4.0])),
            (1, np.array([1.1, 2.1, 3.1, 4.1])),
        ],
        dtype=object,
    )
    np.save(directory / "all_angles.npy", all_angles, allow_pickle=True)
    np.save(directory / "all_surfaces.npy", all_surfaces, allow_pickle=True)
    np.save(directory / "all_popts.npy", all_popts, allow_pickle=True)
    return str(directory)


@pytest.fixture
def binning_result_dir(tmp_path):
    """Create a directory with the log_data_batch_*.txt
    expected by BinningTrajectoryAnalyzer."""
    directory = tmp_path / "binning_result"
    directory.mkdir()
    for idx, (r_eq, zi_c, zi_0, angle) in enumerate(
        [(15.0, 8.0, 6.0, 95.0), (14.5, 7.8, 6.1, 96.5)], start=1
    ):
        path = directory / f"log_data_batch_{idx}.txt"
        path.write_text(
            f"R_eq:{r_eq}\nzi_c:{zi_c}\nzi_0:{zi_0}\nContact angle:{angle}\n"
        )
    return str(directory)


def test_slicing_trajectory_analyzer_load_and_stats(slicing_result_dir):
    analyzer = SlicingTrajectoryAnalyzer(
        [slicing_result_dir], time_steps={slicing_result_dir: 0.5}
    )
    analyzer.load_data()

    surfaces = analyzer.get_surface_areas(slicing_result_dir)
    angles = analyzer.get_contact_angles(slicing_result_dir)
    assert surfaces.shape == (2,)
    assert np.all(surfaces > 0)
    assert angles.shape == (2,)
    assert analyzer.get_method_name() == "Slicing Analysis"


def test_slicing_trajectory_analyzer_plot_evolution(slicing_result_dir, tmp_path):
    analyzer = SlicingTrajectoryAnalyzer([slicing_result_dir])
    analyzer.load_data()

    median_path = tmp_path / "median.png"
    mean_path = tmp_path / "mean.png"
    analyzer.plot_median_angles_evolution(str(median_path))
    analyzer.plot_mean_angles_evolution(str(mean_path))

    assert median_path.exists()
    assert mean_path.exists()


def test_binning_trajectory_analyzer_load(binning_result_dir):
    analyzer = BinningTrajectoryAnalyzer([binning_result_dir])
    analyzer.read_data()  # alias for load_data
    angles = analyzer.get_contact_angles(binning_result_dir)
    surfaces = analyzer.get_surface_areas(binning_result_dir)
    assert angles.shape == (2,)
    assert np.allclose(angles, [95.0, 96.5])
    assert surfaces.shape == (2,)
    assert np.all(surfaces > 0)
    assert analyzer.get_method_name() == "Binning Analysis"


@pytest.mark.parametrize(
    "R,z_center,z_cut,expected",
    [
        (1.0, 0.0, 5.0, 0.0),  # cap entirely above cut
        (1.0, 0.0, -5.0, np.pi),  # cap covers full disk (π·R²)
    ],
)
def test_circular_segment_area_edge_cases(R, z_center, z_cut, expected):
    area = BinningTrajectoryAnalyzer.circular_segment_area(R, z_center, z_cut)
    assert area == pytest.approx(expected, rel=1e-6)


def test_circular_segment_area_partial():
    area = BinningTrajectoryAnalyzer.circular_segment_area(1.0, 0.0, 0.0)
    # Cut at midplane → half disk area
    assert area == pytest.approx(np.pi / 2, rel=1e-6)


def test_circular_segment_area_upper_half():
    # h > R but < 2R: between half and full disk
    area = BinningTrajectoryAnalyzer.circular_segment_area(1.0, 0.0, -0.5)
    full = np.pi
    assert np.pi / 2 < area < full


def test_base_analyze_writes_output_stats(binning_result_dir):
    analyzer = BinningTrajectoryAnalyzer([binning_result_dir])
    analyzer.analyze()
    output_file = os.path.join(binning_result_dir, "output_stats.txt")
    assert os.path.exists(output_file)
    with open(output_file) as f:
        text = f.read()
    assert "Mean Contact Angle:" in text
    assert "Std Contact Angle:" in text
    assert "Mean Surface Area:" in text


def test_base_compute_statistics_and_clean_label(binning_result_dir):
    analyzer = BinningTrajectoryAnalyzer([binning_result_dir])
    analyzer.load_data()
    x, y, yerr = analyzer.compute_statistics(binning_result_dir)
    assert x > 0
    assert y == pytest.approx(np.mean([95.0, 96.5]))
    assert yerr >= 0

    label = analyzer.get_clean_label("result_dump_my_run_reduce_binning")
    assert label == "my_run"


def test_base_plot_mean_angle_vs_surface(binning_result_dir, tmp_path):
    # Two directories with the same fixture content so the linear-fit branch executes.
    second = tmp_path / "binning_result_2"
    second.mkdir()
    for src in os.listdir(binning_result_dir):
        (second / src).write_text(open(os.path.join(binning_result_dir, src)).read())

    analyzer = BinningTrajectoryAnalyzer([binning_result_dir, str(second)])
    save_path = tmp_path / "scaling.png"
    analyzer.plot_mean_angle_vs_surface(save_path=str(save_path))
    assert save_path.exists()


def test_binning_load_files_raises_on_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    analyzer = BinningTrajectoryAnalyzer([str(empty)])
    with pytest.raises(ValueError, match="No log_data_batch"):
        analyzer.load_files()
