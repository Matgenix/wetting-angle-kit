"""End-to-end and branch tests for droplet_slice_plots.py.

Smoke tests confirm the default code paths produce PNG/figure outputs;
the branch tests exercise center=True, molecule_view=True, pbc_y wrapping,
the no-intersection early return, and the ContactAngleAnimator.
"""

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest

from tests.conftest import trajectory_path
from wetting_angle_kit.visualization.droplet_slice_plots import (
    ContactAngleAnimator,
    DropletSlicePlotlyPlotter,
    DropletSlicePlotter,
)


def _synthetic_droplet(seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, 400)
    r = rng.uniform(0.0, 15.0, 400)
    x = r * np.cos(theta) + 50.0
    z = r * np.sin(theta) + 10.0
    y = rng.uniform(0.0, 20.0, 400)
    oxygen = np.column_stack([x, y, z])

    wx = rng.uniform(20.0, 80.0, 150)
    wy = rng.uniform(0.0, 20.0, 150)
    wz = np.zeros(150)
    wall = np.column_stack([wx, wy, wz])

    arc = np.linspace(0, np.pi, 60)
    surface = np.column_stack([50.0 + 14.0 * np.cos(arc), 10.0 + 14.0 * np.sin(arc)])
    return oxygen, wall, [surface], np.array([50.0, 10.0, 14.0, 0.0])


def test_droplet_slicing_plotter_writes_png(tmp_path):
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
    plt.close("all")


def test_droplet_slicing_plotter_plotly_returns_figure():
    """The Plotly version should build a figure with the requested layers."""
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


def test_droplet_slicing_plotter_center_and_molecule_view(tmp_path):
    """center=True + molecule_view=True exercises the recentering and
    water-molecule branches."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    output = tmp_path / "centered_molview.png"
    plotter = DropletSlicePlotter(center=True, show_wall=True, molecule_view=True)
    plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        output_filename=str(output),
        alpha=90.0,
    )
    assert output.exists() and output.stat().st_size > 0


def test_droplet_slicing_plotter_with_pbc_y(tmp_path):
    """pbc_y wrapping branch."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    output = tmp_path / "pbc.png"
    plotter = DropletSlicePlotter(center=False, show_wall=True, molecule_view=False)
    plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        output_filename=str(output),
        alpha=85.0,
        pbc_y=20.0,
    )
    assert output.exists()


def test_droplet_slicing_plotter_no_intersection_returns_early(tmp_path):
    """When discriminant <= 0 the function should close the figure and bail out."""
    oxygen, wall, _, _ = _synthetic_droplet()
    # Surface arc placed high above the wall so z_baseline = min(surf z) is far
    # from z_center → discriminant = radius² - delta_z² < 0.
    arc = np.linspace(0, np.pi, 60)
    high_surface = np.column_stack(
        [50.0 + 14.0 * np.cos(arc), 100.0 + 14.0 * np.sin(arc)]
    )
    popt = np.array([50.0, 10.0, 0.5, 0.0])  # tiny radius, center far below surface
    output = tmp_path / "no_intersect.png"
    plotter = DropletSlicePlotter(center=False, show_wall=True, molecule_view=False)
    plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=[high_surface],
        popt=popt,
        wall_coords=wall,
        output_filename=str(output),
        alpha=120.0,
    )
    # The function bails out before saving — the file should not exist.
    assert not output.exists()


def test_plotly_plotter_center_path():
    """DropletSlicePlotlyPlotter with center=True triggers the recentering branch."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotlyPlotter(center=True)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=85.0,
    )
    assert fig is not None
    assert len(fig.data) >= 5


def test_plotly_plotter_with_pbc_y():
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotlyPlotter(center=False)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=85.0,
        pbc_y=20.0,
    )
    assert len(fig.data) >= 3


def test_plotly_plotter_layers_can_be_disabled():
    """All show_* flags off → figure has zero data traces."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotlyPlotter(center=False)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=None,
        show_water=False,
        show_surface=False,
        show_circle=False,
        show_tangent=False,
        show_wall=False,
    )
    assert len(fig.data) == 0


def test_contact_angle_animator_init_loads_fixture(tmp_path):
    """ContactAngleAnimator.__init__ wires up parsers and finders for a real fixture."""
    animator = ContactAngleAnimator(
        filename=trajectory_path("traj_spherical_drop_4k.lammpstrj"),
        particle_type_wall={3},
        oxygen_type=1,
        hydrogen_type=2,
        liquid_particle_types={1, 2},
        n_frames=1,
        droplet_geometry="cylinder_y",
        delta_cylinder=20,
        max_dist=50,
        width_cylinder=20,
    )
    assert animator.wall_coords.shape[1] == 3
    assert animator.oxygen_indices.size > 0
    assert animator.parser is not None
    assert animator.plotter is not None


@pytest.mark.slow
def test_contact_angle_animator_generates_html(tmp_path):
    """Smoke-test ContactAngleAnimator on the LAMMPS fixture via cylinder_y geometry."""
    output = tmp_path / "animation.html"
    animator = ContactAngleAnimator(
        filename=trajectory_path("traj_spherical_drop_4k.lammpstrj"),
        particle_type_wall={3},
        oxygen_type=1,
        hydrogen_type=2,
        liquid_particle_types={1, 2},
        n_frames=1,
        droplet_geometry="cylinder_y",
        delta_cylinder=20,
        max_dist=50,
        width_cylinder=20,
    )
    animator.generate_animation(output_filename=str(output))
    assert output.exists()
    assert output.stat().st_size > 0
