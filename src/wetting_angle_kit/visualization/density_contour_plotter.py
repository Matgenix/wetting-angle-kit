"""Density-field contour plot with the fitted spherical cap overlay.

Mirrors the visuals of the legacy
``BinningTrajectoryPlotter.plot_density_contour`` (Jet colormap,
dashed cap arc, dotted wall line, equal x/y aspect) while accepting
the new coupled-binning result shapes:

- :class:`CoupledBinning2DBatchResult` — single batch, plotted directly.
- :class:`CoupledBinning2DResults` — densities averaged across batches.
- :class:`CoupledBinning3DBatchResult` — 3D density azimuthally
  averaged on the ``(xi, yi)`` plane to a 2D ``(r, zi)`` field.
- :class:`CoupledBinning3DResults` — averaged across batches first,
  then azimuthally collapsed.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from wetting_angle_kit.analysis.results import (
    CoupledBinning2DBatchResult,
    CoupledBinning2DResults,
    CoupledBinning3DBatchResult,
    CoupledBinning3DResults,
)


class DensityContourPlotter:
    """Plot a binned density field with the fitted cap and wall overlaid.

    Parameters
    ----------
    source
        Single batch or full results object as listed in the module
        docstring.
    label : str, default ``"trajectory"``
        Display label used in the figure title.
    colorscale : str, default ``"Jet"``
        Plotly colorscale for the density contour. The legacy plotter
        used ``"Jet"``; the default is kept for visual continuity.
    """

    def __init__(
        self,
        source: Any,
        *,
        label: str = "trajectory",
        colorscale: str = "Jet",
    ) -> None:
        self.source = source
        self.label = label
        self.colorscale = colorscale

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        title: str | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        """Build the density contour figure.

        Parameters
        ----------
        title : str, optional
            Figure title. Defaults to a ``"Density field — {label}
            (batch_descriptor)"`` string, where the batch descriptor
            names the batch when the source is a single batch and
            ``"averaged"`` when it is a full results object.
        save_path : str, optional
            If provided, write the figure to standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            Contour + dashed fitted cap + dotted wall line.
        """
        (
            xi,
            zi,
            density,
            model_params,
            batch_descriptor,
        ) = self._extract(self.source)

        dxi = xi[-1] - xi[-2] if len(xi) >= 2 else 0.0
        xi_lo = float(xi[0] - dxi / 2)
        xi_hi = float(xi[-1] + dxi / 2)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=xi,
                y=zi,
                z=density.T,
                colorscale=self.colorscale,
                name="Liquid density",
                colorbar={
                    "title": {"text": "ρ", "font": {"size": 16}},
                    "tickfont": {"size": 14},
                    "len": 0.75,
                    "y": 0,
                    "yanchor": "bottom",
                },
            )
        )

        circle_xi, circle_zi, wall_xi, wall_zi = self._cap_and_wall_traces(
            model_params, xi_lo, xi_hi
        )
        if circle_xi.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=circle_xi,
                    y=circle_zi,
                    mode="lines",
                    name="Fitted droplet",
                    line={"color": "black", "dash": "dash", "width": 2},
                )
            )
        fig.add_trace(
            go.Scatter(
                x=wall_xi,
                y=wall_zi,
                mode="lines",
                name="Fitted wall",
                line={"color": "black", "dash": "dot", "width": 2},
            )
        )

        default_title = f"Density field — {self.label}"
        if batch_descriptor:
            default_title += f" ({batch_descriptor})"
        fig.update_layout(
            title=title or default_title,
            template="plotly_white",
            xaxis={
                "title": {"text": "ξ (Å)", "font": {"size": 16}},
                "tickfont": {"size": 14},
                "range": [xi_lo, xi_hi],
                "constrain": "domain",
            },
            yaxis={
                "title": {"text": "z (Å)", "font": {"size": 16}},
                "tickfont": {"size": 14},
                "scaleanchor": "x",
                "scaleratio": 1,
                "constrain": "domain",
            },
            legend={"x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top"},
        )
        if save_path:
            fig.write_html(save_path)
        return fig

    # ------------------------------------------------------------------
    # Internals — source dispatch.
    # ------------------------------------------------------------------

    def _extract(
        self, source: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, str]:
        if isinstance(source, CoupledBinning2DBatchResult):
            return (
                source.xi_grid,
                source.zi_grid,
                source.density,
                source.model_params,
                "",
            )
        if isinstance(source, CoupledBinning2DResults):
            if not source.batches:
                raise ValueError("CoupledBinning2DResults has no batches to plot.")
            ref2d = source.batches[0]
            densities = np.stack([b.density for b in source.batches], axis=0)
            mean_density = densities.mean(axis=0)
            return (
                ref2d.xi_grid,
                ref2d.zi_grid,
                mean_density,
                ref2d.model_params,
                f"averaged over {len(source.batches)} batches",
            )
        if isinstance(source, CoupledBinning3DBatchResult):
            xi, zi, density2d = self._azimuthal_average_3d(
                source.xi_grid,
                source.yi_grid,
                source.zi_grid,
                source.density,
            )
            return (
                xi,
                zi,
                density2d,
                source.model_params,
                "azimuthally averaged",
            )
        if isinstance(source, CoupledBinning3DResults):
            if not source.batches:
                raise ValueError("CoupledBinning3DResults has no batches to plot.")
            ref3d: CoupledBinning3DBatchResult = source.batches[0]
            densities = np.stack([b.density for b in source.batches], axis=0)
            mean_density = densities.mean(axis=0)
            xi, zi, density2d = self._azimuthal_average_3d(
                ref3d.xi_grid,
                ref3d.yi_grid,
                ref3d.zi_grid,
                mean_density,
            )
            return (
                xi,
                zi,
                density2d,
                ref3d.model_params,
                f"averaged over {len(source.batches)} batches, azimuthally averaged",
            )
        raise TypeError(
            f"DensityContourPlotter does not know how to plot {type(source).__name__}."
        )

    # ------------------------------------------------------------------
    # Internals — 3D → 2D azimuthal average.
    # ------------------------------------------------------------------

    @staticmethod
    def _azimuthal_average_3d(
        xi_cc: np.ndarray,
        yi_cc: np.ndarray,
        zi_cc: np.ndarray,
        density: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collapse ``density(xi, yi, zi)`` onto ``density(r, zi)``."""
        XI, YI = np.meshgrid(xi_cc, yi_cc, indexing="ij")
        r_flat = np.sqrt(XI**2 + YI**2).ravel()
        r_max = float(r_flat.max())
        n_r = min(len(xi_cc), len(yi_cc))
        r_edges = np.linspace(0.0, r_max, n_r + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        bin_idx = np.clip(
            np.searchsorted(r_edges, r_flat, side="right") - 1, 0, n_r - 1
        )
        density2d = np.zeros((n_r, len(zi_cc)))
        for k in range(len(zi_cc)):
            slice_flat = density[:, :, k].ravel()
            sums = np.bincount(bin_idx, weights=slice_flat, minlength=n_r)
            counts = np.bincount(bin_idx, minlength=n_r)
            with np.errstate(invalid="ignore", divide="ignore"):
                density2d[:, k] = np.where(counts > 0, sums / counts, 0.0)
        return r_centers, zi_cc, density2d

    # ------------------------------------------------------------------
    # Internals — cap arc + wall line geometry.
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_and_wall_traces(
        model_params: dict, xi_lo: float, xi_hi: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the fitted spherical-cap arc and the wall-line trace."""
        R_eq = float(model_params["R_eq"])
        zi_c = float(model_params["zi_c"])
        zi_0 = float(model_params["zi_0"])
        discriminant = R_eq**2 - (zi_0 - zi_c) ** 2
        if discriminant < 0:
            cap_xi = np.array([])
            cap_zi = np.array([])
        else:
            xi_cross = float(np.sqrt(discriminant))
            alpha_inf = np.arctan((zi_0 - zi_c) / xi_cross)
            alpha = np.linspace(alpha_inf, np.pi / 2, 200)
            cap_xi = R_eq * np.cos(alpha)
            cap_zi = zi_c + R_eq * np.sin(alpha)
        wall_xi = np.array([xi_lo, xi_hi])
        wall_zi = np.array([zi_0, zi_0])
        return cap_xi, cap_zi, wall_xi, wall_zi
