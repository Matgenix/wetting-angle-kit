"""Smoke tests for the visualization helpers.

These ensure plotting code paths run end-to-end and write non-empty PNGs.
They do not validate pixel-level rendering; they only catch regressions
that break the call graph (wrong kwargs, missing layers, etc.).
"""

from __future__ import annotations

# Force the non-interactive backend before any pyplot import in this module.
import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402


def _synthetic_droplet(seed=0):
    """Return synthetic (oxygen_positions, wall_coords, surface_data, popt)
    that resemble what the sliced pipeline produces."""
    rng = np.random.default_rng(seed)
    # Oxygen positions: a half-shell-ish cloud above a wall.
    theta = rng.uniform(0, np.pi, 600)
    r = rng.uniform(0.0, 15.0, 600)
    x = r * np.cos(theta) + 50.0
    z = r * np.sin(theta) + 10.0
    y = rng.uniform(0.0, 20.0, 600)
    oxygen = np.column_stack([x, y, z])

    # Wall: flat layer at z=0.
    wx = rng.uniform(20.0, 80.0, 200)
    wy = rng.uniform(0.0, 20.0, 200)
    wz = np.zeros(200)
    wall = np.column_stack([wx, wy, wz])

    # Surface contour (single slice): a half-circle in XZ.
    arc = np.linspace(0, np.pi, 60)
    surface = np.column_stack(
        [
            50.0 + 14.0 * np.cos(arc),
            10.0 + 14.0 * np.sin(arc),
        ]
    )
    surface_data = [surface]

    # Circle parameters [Xc, Zc, R, baseline_z]
    popt = np.array([50.0, 10.0, 14.0, 0.0])
    return oxygen, wall, surface_data, popt


def test_droplet_sliced_plotter_writes_png(tmp_path):
    from wetting_angle_kit.visualization import DropletSlicePlotter

    oxygen, wall, surface_data, popt = _synthetic_droplet()
    output = tmp_path / "droplet.png"
    plotter = DropletSlicePlotter(center=False, show_wall=True, molecule_view=False)
    plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        output_filename=str(output),
        alpha=90.0,
    )
    assert output.exists()
    assert output.stat().st_size > 0
    # Be tidy about state across tests.
    plt.close("all")


def test_droplet_sliced_plotter_plotly_returns_figure():
    """The Plotly version should build a figure with the requested layers."""
    import plotly.graph_objects as go
    from wetting_angle_kit.visualization import DropletSlicePlotlyPlotter

    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotlyPlotter(center=False)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=90.0,
    )
    assert isinstance(fig, go.Figure)
    # At least the wall, water, surface, circle, tangent, and arc traces.
    assert len(fig.data) >= 5


def test_plot_liquid_particles_returns_axes():
    from wetting_angle_kit.visualization import plot_liquid_particles

    rng = np.random.default_rng(1)
    positions = rng.normal(size=(50, 3))
    ax = plot_liquid_particles(positions)
    assert ax is not None
    # Subsampling path
    ax2 = plot_liquid_particles(positions, subsample=10)
    assert ax2 is not None
    plt.close("all")


def test_read_surface_file_handles_two_and_three_columns(tmp_path):
    from wetting_angle_kit.visualization import read_surface_file

    two_col = tmp_path / "two.txt"
    two_col.write_text("1.0 2.0\n3.0 4.0\n")
    x, y, z = read_surface_file(str(two_col))
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [2.0, 4.0])
    np.testing.assert_array_equal(z, [0.0, 0.0])

    three_col = tmp_path / "three.txt"
    three_col.write_text("1.0 2.0 0.5\n3.0 4.0 0.6\n")
    x, y, z = read_surface_file(str(three_col))
    np.testing.assert_array_equal(z, [0.5, 0.6])


# --- compute_isoline produces sane output for a well-formed fit ---


def test_hyperbolic_tangent_compute_isoline_well_formed():
    """Wall inside the fitted sphere should yield finite isoline arrays."""
    from wetting_angle_kit.contact_angle_methods.binning.surface_definition import (
        HyperbolicTangentModel,
    )

    model = HyperbolicTangentModel()
    # Wall at z=0, droplet center at z=8, radius 10 → wall is inside the
    # sphere (|0 - 8| = 8 < 10). Densities and thicknesses are positive.
    model.params = [1.0, 0.0, 10.0, 8.0, 0.0, 1.0, 1.0]
    circle_xi, circle_zi, wall_xi, wall_zi = model.compute_isoline()
    assert circle_xi.size == 100
    assert np.all(np.isfinite(circle_xi))
    assert np.all(np.isfinite(circle_zi))
    assert np.all(np.isfinite(wall_xi))
    np.testing.assert_allclose(wall_zi, 0.0)
