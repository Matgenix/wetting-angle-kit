import matplotlib

matplotlib.use("Agg", force=False)

import numpy as np

from wetting_angle_kit.visualization.surface_plots import (
    plot_liquid_particles,
    plot_slice,
    plot_surface_and_points,
    plot_surface_file,
    read_surface_file,
    visualize_surface_with_points,
)


def _write_surface(tmp_path, columns):
    path = tmp_path / "surface.dat"
    np.savetxt(path, columns)
    return str(path)


def test_plot_surface_file_returns_xy(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    path = _write_surface(tmp_path, data)
    x, y = plot_surface_file(path)
    assert np.allclose(x, [1.0, 3.0, 5.0])
    assert np.allclose(y, [2.0, 4.0, 6.0])


def test_read_surface_file_two_columns_pads_z(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = _write_surface(tmp_path, data)
    x, y, z = read_surface_file(path)
    assert np.allclose(x, [1.0, 3.0])
    assert np.allclose(y, [2.0, 4.0])
    assert np.allclose(z, [0.0, 0.0])


def test_read_surface_file_three_columns(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = _write_surface(tmp_path, data)
    x, y, z = read_surface_file(path)
    assert np.allclose(x, [1.0, 4.0])
    assert np.allclose(y, [2.0, 5.0])
    assert np.allclose(z, [3.0, 6.0])


def test_plot_slice_runs():
    plot_slice(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]))


def test_plot_surface_and_points_runs():
    x = np.linspace(0, 1, 5)
    plot_surface_and_points(x, x, x, x + 0.1, x + 0.2, x + 0.3)


def test_visualize_surface_with_points_runs(tmp_path):
    data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    path = _write_surface(tmp_path, data)
    points = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
    visualize_surface_with_points(path, points)


def test_plot_liquid_particles_creates_axes():
    positions = np.random.default_rng(0).uniform(size=(50, 3))
    ax = plot_liquid_particles(positions)
    assert ax is not None


def test_plot_liquid_particles_subsample():
    positions = np.random.default_rng(0).uniform(size=(100, 3))
    ax = plot_liquid_particles(positions, subsample=10)
    assert ax is not None


def test_plot_liquid_particles_uses_given_ax():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    positions = np.random.default_rng(0).uniform(size=(20, 3))
    returned = plot_liquid_particles(positions, ax=ax)
    assert returned is ax
