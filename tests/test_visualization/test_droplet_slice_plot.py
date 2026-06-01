"""Smoke tests for the plotly droplet-slice plotter and the animator."""

import numpy as np
import plotly.graph_objects as go
import pytest

from tests.conftest import trajectory_path
from wetting_angle_kit.visualization import DropletSlicePlotter
from wetting_angle_kit.visualization.animator import ContactAngleAnimator


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


# --- DropletSlicePlotter (plotly) ---


def test_droplet_slice_plotter_returns_figure():
    """Default code path builds a plotly figure with the expected layers."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotter(center=False)
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


def test_droplet_slice_plotter_center_path():
    """center=True triggers the recentering branch."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotter(center=True)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=85.0,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 5


def test_droplet_slice_plotter_with_pbc_y():
    """pbc_y wrapping branch."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotter(center=False)
    fig = plotter.plot_surface_points(
        oxygen_position=oxygen,
        surface_data=surface_data,
        popt=popt,
        wall_coords=wall,
        alpha=85.0,
        pbc_y=20.0,
    )
    assert len(fig.data) >= 3


def test_droplet_slice_plotter_layers_can_be_disabled():
    """All show_* flags off → figure has zero data traces."""
    oxygen, wall, surface_data, popt = _synthetic_droplet()
    plotter = DropletSlicePlotter(center=False)
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


# --- ContactAngleAnimator (not re-exported; import from submodule) ---


def test_contact_angle_animator_init_loads_fixture():
    """ContactAngleAnimator.__init__ wires up parsers and finders for a real fixture."""
    pytest.importorskip("ovito")
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
    """Smoke-test ContactAngleAnimator on the cylindrical LAMMPS fixture."""
    pytest.importorskip("ovito")
    output = tmp_path / "animation.html"
    animator = ContactAngleAnimator(
        filename=trajectory_path("traj_10_3_330w_nve_4k_reajust.lammpstrj"),
        particle_type_wall={3},
        oxygen_type=1,
        hydrogen_type=2,
        liquid_particle_types={1, 2},
        n_frames=1,
        droplet_geometry="cylinder_y",
        delta_cylinder=20,
        max_dist=50,
        width_cylinder=21,
    )
    animator.generate_animation(output_filename=str(output))
    assert output.exists()
    assert output.stat().st_size > 0
