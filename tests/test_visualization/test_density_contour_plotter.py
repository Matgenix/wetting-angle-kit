"""Smoke tests for :class:`DensityContourPlotter`."""

import numpy as np
import plotly.graph_objects as go
import pytest

from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit2DResults,
    CoupledFit3DBatchResult,
    CoupledFit3DResults,
)
from wetting_angle_kit.visualization import DensityContourPlotter


def _model_params_2d() -> dict:
    return {
        "rho1": 0.03,
        "rho2": 1e-4,
        "R_eq": 25.0,
        "zi_c": 5.0,
        "zi_0": 5.0,
        "t1": 1.0,
        "t2": 1.0,
    }


def _model_params_3d() -> dict:
    return {
        "rho1": 0.03,
        "rho2": 1e-4,
        "R_eq": 25.0,
        "xi_c": 0.0,
        "yi_c": 0.0,
        "zi_c": 5.0,
        "zi_0": 5.0,
        "t1": 1.0,
        "t2": 1.0,
    }


def _make_2d_batch(seed: int = 0) -> CoupledFit2DBatchResult:
    rng = np.random.default_rng(seed)
    xi = np.linspace(0.0, 40.0, 15)
    zi = np.linspace(0.0, 40.0, 15)
    density = rng.uniform(0.0, 0.03, size=(15, 15))
    return CoupledFit2DBatchResult(
        frames=[0, 1],
        angle=95.0,
        model_params=_model_params_2d(),
        xi_grid=xi,
        zi_grid=zi,
        density=density,
    )


def _make_3d_batch() -> CoupledFit3DBatchResult:
    xi = np.linspace(-30.0, 30.0, 10)
    yi = np.linspace(-30.0, 30.0, 10)
    zi = np.linspace(0.0, 35.0, 12)
    XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing="ij")
    r = np.sqrt(XI**2 + YI**2 + (ZI - 5.0) ** 2)
    density = (
        0.5
        * (0.03 + 1e-4 - (0.03 - 1e-4) * np.tanh(2 * (r - 25.0) / 1.0))
        * 0.5
        * (1.0 + np.tanh(2 * (ZI - 5.0) / 1.0))
    )
    return CoupledFit3DBatchResult(
        frames=[0],
        angle=90.0,
        model_params=_model_params_3d(),
        xi_grid=xi,
        yi_grid=yi,
        zi_grid=zi,
        density=density,
    )


# ----------------------------- 2D --------------------------------------------


def test_density_contour_plotter_2d_batch_runs() -> None:
    fig = DensityContourPlotter(_make_2d_batch(), label="run-A").plot()
    assert isinstance(fig, go.Figure)
    # Contour + cap + wall ⇒ 3 traces.
    assert len(fig.data) == 3
    names = {getattr(t, "name", None) for t in fig.data}
    assert {"Liquid density", "Fitted droplet", "Fitted wall"} <= names
    # Title carries the label but does NOT include the frame list
    # (which can be long for pooled batches).
    title_text = fig.layout.title.text
    assert "run-A" in title_text
    assert "frames" not in title_text
    assert "[0, 1]" not in title_text
    # No empty trailing parenthesis either.
    assert not title_text.rstrip().endswith("()")


def test_density_contour_plotter_2d_results_averages_density() -> None:
    b1 = _make_2d_batch(seed=0)
    b2 = _make_2d_batch(seed=1)
    results = CoupledFit2DResults(batches=[b1, b2], method_metadata={})
    fig = DensityContourPlotter(results).plot()
    assert isinstance(fig, go.Figure)
    contour_z = np.array(fig.data[0].z)
    expected = 0.5 * (b1.density + b2.density)
    np.testing.assert_allclose(contour_z, expected.T, atol=1e-12)
    assert "averaged over 2 batches" in fig.layout.title.text


def test_density_contour_plotter_2d_empty_results_raises() -> None:
    results = CoupledFit2DResults(batches=[], method_metadata={})
    with pytest.raises(ValueError, match="no batches"):
        DensityContourPlotter(results).plot()


def test_density_contour_plotter_legacy_visuals() -> None:
    """Cap is dashed black, wall is dotted black, colorbar shows ρ."""
    fig = DensityContourPlotter(_make_2d_batch()).plot()
    contour, cap, wall = fig.data
    assert cap.line.dash == "dash"
    assert wall.line.dash == "dot"
    assert cap.line.color == "black"
    assert wall.line.color == "black"
    # Colorbar title preserves the legacy ρ glyph.
    assert contour.colorbar.title.text == "ρ"
    # Equal x/y aspect ratio is preserved.
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


# ----------------------------- 3D --------------------------------------------


def test_density_contour_plotter_3d_batch_runs() -> None:
    fig = DensityContourPlotter(_make_3d_batch()).plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    contour_x = np.array(fig.data[0].x)
    assert contour_x.min() >= 0.0  # r ≥ 0
    assert "azimuthally averaged" in fig.layout.title.text


def test_density_contour_plotter_3d_results_runs() -> None:
    results = CoupledFit3DResults(
        batches=[_make_3d_batch(), _make_3d_batch()], method_metadata={}
    )
    fig = DensityContourPlotter(results).plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_density_contour_plotter_unknown_source_raises() -> None:
    with pytest.raises(TypeError, match="does not know how to plot"):
        DensityContourPlotter("not a results object").plot()
