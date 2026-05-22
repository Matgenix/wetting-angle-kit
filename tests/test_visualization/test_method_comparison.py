import os

import pytest

from wetting_angle_kit.visualization.binning_trajectory_analyzer import (
    BinningTrajectoryAnalyzer,
)
from wetting_angle_kit.visualization.method_comparison import MethodComparison


def _write_binning_dir(directory, samples):
    directory.mkdir()
    for idx, (r_eq, zi_c, zi_0, angle) in enumerate(samples, start=1):
        (directory / f"log_data_batch_{idx}.txt").write_text(
            f"R_eq:{r_eq}\nzi_c:{zi_c}\nzi_0:{zi_0}\nContact angle:{angle}\n"
        )


@pytest.fixture
def two_binning_analyzers(tmp_path):
    d1 = tmp_path / "result_dump_runA"
    d2 = tmp_path / "result_dump_runB"
    _write_binning_dir(d1, [(15.0, 8.0, 6.0, 95.0), (14.5, 7.8, 6.1, 96.5)])
    _write_binning_dir(d2, [(16.0, 8.2, 6.0, 92.0), (15.5, 8.0, 6.1, 93.5)])

    analyzer1 = BinningTrajectoryAnalyzer([str(d1)])
    analyzer2 = BinningTrajectoryAnalyzer([str(d2)])
    analyzer1.load_data()
    analyzer2.load_data()
    # Generate output_stats.txt for both
    analyzer1.analyze()
    analyzer2.analyze()
    return analyzer1, analyzer2


def test_method_comparison_compare_statistics(two_binning_analyzers):
    a1, a2 = two_binning_analyzers
    comparator = MethodComparison([a1, a2], method_names=["Binning A", "Binning B"])
    report = comparator.compare_statistics()
    assert "METHOD COMPARISON STATISTICS" in report
    assert "Binning A" in report
    assert "Binning B" in report
    assert "Mean Angle" in report
    assert "Std Angle" in report


def test_method_comparison_default_method_names(two_binning_analyzers):
    a1, a2 = two_binning_analyzers
    comparator = MethodComparison([a1, a2])
    assert comparator.method_names == ["Binning Analysis", "Binning Analysis"]


def test_method_comparison_side_by_side_plot(two_binning_analyzers, tmp_path):
    a1, a2 = two_binning_analyzers
    comparator = MethodComparison([a1, a2], method_names=["A", "B"])
    save_path = tmp_path / "side_by_side.png"
    comparator.plot_side_by_side_comparison(save_path=str(save_path))
    assert save_path.exists()


def test_method_comparison_overlay_plot(two_binning_analyzers, tmp_path):
    a1, a2 = two_binning_analyzers
    comparator = MethodComparison([a1, a2], method_names=["A", "B"])
    save_path = tmp_path / "overlay.png"
    comparator.plot_overlay_comparison(save_path=str(save_path))
    assert save_path.exists()


def test_method_comparison_side_by_side_single_analyzer(
    two_binning_analyzers, tmp_path
):
    # Cover the `len(self.analyzers) == 1` branch
    a1, _ = two_binning_analyzers
    comparator = MethodComparison([a1])
    save_path = tmp_path / "single.png"
    comparator.plot_side_by_side_comparison(save_path=str(save_path))
    assert save_path.exists()


def test_method_comparison_check_and_run_raises_when_missing(tmp_path):
    """_check_and_run_analysis raises if output_stats.txt is absent."""
    d = tmp_path / "result_dump_empty"
    d.mkdir()
    (d / "log_data_batch_1.txt").write_text(
        "R_eq:15.0\nzi_c:8.0\nzi_0:6.0\nContact angle:90.0\n"
    )
    analyzer = BinningTrajectoryAnalyzer([str(d)])
    analyzer.load_data()
    # do not call analyze(), so output_stats.txt is missing
    comparator = MethodComparison([analyzer])
    with pytest.raises(FileNotFoundError, match="No analysis found"):
        comparator._check_and_run_analysis(analyzer)


def test_method_comparison_compare_statistics_falls_back_when_stats_missing(tmp_path):
    """compare_statistics() recovers via in-memory data
    if output_stats.txt is missing."""
    d = tmp_path / "result_dump_no_stats"
    _write_binning_dir(d, [(15.0, 8.0, 6.0, 95.0), (14.5, 7.8, 6.1, 96.5)])
    analyzer = BinningTrajectoryAnalyzer([str(d)])
    analyzer.load_data()
    # Remove output_stats.txt path is never created (analyze() not called)
    assert not os.path.exists(os.path.join(str(d), "output_stats.txt"))
    comparator = MethodComparison([analyzer])
    report = comparator.compare_statistics()
    assert "Mean Angle" in report
