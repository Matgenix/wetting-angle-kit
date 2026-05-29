import numpy as np
import plotly.graph_objects as go
import pytest

from wetting_angle_kit.analysis.binning.results import BinningBatch, BinningResults
from wetting_angle_kit.analysis.slicing.results import SlicingResults
from wetting_angle_kit.visualization.binning_trajectory_plotter import (
    BinningTrajectoryPlotter,
)
from wetting_angle_kit.visualization.slicing_trajectory_plotter import (
    SlicingTrajectoryPlotter,
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
def slicing_results():
    polygon = _square_polygon(side=4.0)
    return SlicingResults(
        frames=[0, 1],
        angles=[
            np.array([85.0, 90.0, 95.0]),
            np.array([87.0, 92.0, 96.0]),
        ],
        surfaces=[
            [polygon, polygon * 1.1],
            [polygon * 1.05, polygon * 1.15],
        ],
        popts=[
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1]),
        ],
    )


@pytest.fixture
def binning_results():
    return BinningResults(
        batches=[
            BinningBatch(
                batch_index=1,
                angle=95.0,
                n_particles=100.0,
                xi_cc=np.linspace(0.0, 10.0, 5),
                zi_cc=np.linspace(0.0, 10.0, 5),
                rho_cc=np.ones((5, 5)),
                circle_xi=np.array([0.0, 1.0, 2.0]),
                circle_zi=np.array([5.0, 6.0, 7.0]),
                wall_line_xi=np.array([0.0, 1.0, 2.0]),
                wall_line_zi=np.array([6.0, 6.0, 6.0]),
                fitted_params={"R_eq": 15.0, "zi_c": 8.0, "zi_0": 6.0},
            ),
            BinningBatch(
                batch_index=2,
                angle=96.5,
                n_particles=110.0,
                xi_cc=np.linspace(0.0, 10.0, 5),
                zi_cc=np.linspace(0.0, 10.0, 5),
                rho_cc=np.ones((5, 5)),
                circle_xi=None,
                circle_zi=None,
                wall_line_xi=None,
                wall_line_zi=None,
                fitted_params={"R_eq": 14.5, "zi_c": 7.8, "zi_0": 6.1},
            ),
        ]
    )


# --- SlicingTrajectoryPlotter ---


def test_slicing_plotter_summary(slicing_results):
    plotter = SlicingTrajectoryPlotter(slicing_results, labels=["A"])
    [stats] = plotter.summary()
    assert stats.method_name == "Slicing Analysis"
    assert stats.label == "A"
    assert stats.n_samples == 2
    # mean of per-frame means: mean([90.0, 91.667]) ≈ 90.83
    assert 80.0 < stats.mean_contact_angle < 100.0
    assert stats.mean_surface_area > 0


def test_slicing_plotter_plot_angle_evolution_returns_figure(slicing_results):
    plotter = SlicingTrajectoryPlotter(slicing_results, time_steps=[0.5])
    fig = plotter.plot_angle_evolution(stat="median")
    assert isinstance(fig, go.Figure)
    fig_mean = plotter.plot_angle_evolution(stat="mean")
    assert isinstance(fig_mean, go.Figure)


def test_slicing_plotter_rejects_unknown_stat(slicing_results):
    plotter = SlicingTrajectoryPlotter(slicing_results)
    with pytest.raises(ValueError, match="stat must be"):
        plotter.plot_angle_evolution(stat="bogus")


# --- BinningTrajectoryPlotter ---


def test_binning_plotter_summary(binning_results):
    plotter = BinningTrajectoryPlotter(binning_results, labels=["A"])
    [stats] = plotter.summary()
    assert stats.method_name == "Binning Analysis"
    assert stats.label == "A"
    assert stats.n_samples == 2
    assert stats.mean_contact_angle == pytest.approx(np.mean([95.0, 96.5]))
    assert stats.std_contact_angle == pytest.approx(np.std([95.0, 96.5]))
    assert stats.mean_surface_area > 0


def test_binning_plotter_summary_str_block(binning_results):
    plotter = BinningTrajectoryPlotter(binning_results)
    [stats] = plotter.summary()
    text = str(stats)
    assert "Mean Contact Angle:" in text
    assert "Std Contact Angle:" in text
    assert "Mean Surface Area:" in text


def test_binning_plotter_plot_angle_evolution_returns_figure(binning_results):
    plotter = BinningTrajectoryPlotter(binning_results, time_steps=[2.0])
    fig = plotter.plot_angle_evolution()
    assert isinstance(fig, go.Figure)


def test_binning_plotter_density_contour_with_isoline(binning_results):
    plotter = BinningTrajectoryPlotter(binning_results)
    fig = plotter.plot_density_contour(batch_index=0)
    assert isinstance(fig, go.Figure)
    # contour + circle + wall = 3 traces
    assert len(fig.data) == 3


def test_binning_plotter_density_contour_without_isoline(binning_results):
    plotter = BinningTrajectoryPlotter(binning_results)
    # second batch has circle/wall = None
    fig = plotter.plot_density_contour(batch_index=1)
    assert isinstance(fig, go.Figure)
    # only the contour trace when isoline is missing
    assert len(fig.data) == 1


# --- circular_segment_area static method ---


@pytest.mark.parametrize(
    "R,z_center,z_cut,expected",
    [
        (1.0, 0.0, 5.0, 0.0),  # cap entirely above cut
        (1.0, 0.0, -5.0, np.pi),  # cap covers full disk (π·R²)
    ],
)
def test_circular_segment_area_edge_cases(R, z_center, z_cut, expected):
    area = BinningTrajectoryPlotter.circular_segment_area(R, z_center, z_cut)
    assert area == pytest.approx(expected, rel=1e-6)


def test_circular_segment_area_partial():
    area = BinningTrajectoryPlotter.circular_segment_area(1.0, 0.0, 0.0)
    # Cut at midplane → half disk area
    assert area == pytest.approx(np.pi / 2, rel=1e-6)


def test_circular_segment_area_upper_half():
    # h > R but < 2R: between half and full disk
    area = BinningTrajectoryPlotter.circular_segment_area(1.0, 0.0, -0.5)
    full = np.pi
    assert np.pi / 2 < area < full
