"""Density-field contour plot with the fitted spherical cap overlay.

Renders a 2D density contour with the fitted cap arc (dashed) and
wall line (dotted) overlaid (equal x/y aspect, Jet colormap by
default). Accepts any of the coupled-fit result types:

- :class:`CoupledFit2DBatchResult` — single batch, plotted directly.
- :class:`CoupledFit2DResults` — densities averaged across batches.
- :class:`CoupledFit3DBatchResult` — 3D density azimuthally
  averaged on the ``(xi, yi)`` plane to a 2D ``(r, zi)`` field.
- :class:`CoupledFit3DResults` — averaged across batches first,
  then azimuthally collapsed.
"""

from typing import Any

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from wetting_angle_kit.analysis.results import (
    CoupledFit2DBatchResult,
    CoupledFit2DResults,
    CoupledFit3DBatchResult,
    CoupledFit3DResults,
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
        Plotly colorscale for the density contour.
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
    # 3D isosurface plot with density-threshold slider.
    # ------------------------------------------------------------------

    def plot_3d_isosurface(
        self,
        *,
        n_levels: int = 10,
        show_fit: bool = True,
        title: str | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        """3D isosurface of the density field with a density-threshold slider.

        Only accepts :class:`CoupledFit3DBatchResult` or
        :class:`CoupledFit3DResults` sources (the full 3D density grid
        is required).

        Parameters
        ----------
        n_levels : int, default 10
            Number of iso-density levels exposed in the slider.
        show_fit : bool, default True
            If ``True``, overlay the fitted sphere as a black dashed
            wireframe (meridians + parallels).
        title : str, optional
            Figure title.  Defaults to
            ``"Isosurface ρ ≥ ... — {label}"``.
        save_path : str, optional
            If provided, write the figure to standalone HTML.

        Returns
        -------
        plotly.graph_objects.Figure
            3D isosurface with a wall plane and a density slider.
        """
        xi, yi, zi, density, params = self._extract_3d(self.source)

        XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing="ij")
        positive = density[density > 0]
        rho_min = float(positive.min()) if positive.size > 0 else 0.0
        rho_max = float(density.max())
        iso_levels = np.linspace(
            rho_min + 0.05 * (rho_max - rho_min),
            0.95 * rho_max,
            n_levels,
        )

        # Sample one color per iso-level from the colorscale.
        t_values = [
            (iso_val - rho_min) / (rho_max - rho_min) if rho_max > rho_min else 0.5
            for iso_val in iso_levels
        ]
        iso_colors = pc.sample_colorscale(self.colorscale, t_values)

        fig = go.Figure()
        for i, iso_val in enumerate(iso_levels):
            color = iso_colors[i]
            # Single-color colorscale so the entire isosurface is uniform.
            uniform_cs = [[0, color], [1, color]]
            fig.add_trace(
                go.Isosurface(
                    x=XI.ravel(),
                    y=YI.ravel(),
                    z=ZI.ravel(),
                    value=density.ravel(),
                    isomin=float(iso_val),
                    isomax=float(rho_max),
                    surface_count=1,
                    caps={"x_show": False, "y_show": False, "z_show": False},
                    colorscale=uniform_cs,
                    visible=(i == 0),
                    opacity=0.6,
                    showscale=False,
                )
            )

        # Semi-transparent wall plane.
        z0 = float(params["zi_0"])
        wall_x = np.array([[xi[0], xi[-1]], [xi[0], xi[-1]]])
        wall_y = np.array([[yi[0], yi[0]], [yi[-1], yi[-1]]])
        wall_z = np.full_like(wall_x, z0)
        fig.add_trace(
            go.Surface(
                x=wall_x,
                y=wall_y,
                z=wall_z,
                colorscale=[
                    [0, "rgba(0,0,0,0.15)"],
                    [1, "rgba(0,0,0,0.15)"],
                ],
                showscale=False,
                name="Wall plane",
            )
        )

        # Reference colorbar: an invisible Scatter3d that carries the
        # full-range Jet colorbar so the user sees where the current
        # iso-level sits on the density scale.
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={
                    "size": 0,
                    "color": [rho_min, rho_max],
                    "colorscale": self.colorscale,
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": "ρ", "font": {"size": 16}},
                        "tickfont": {"size": 14},
                        "len": 0.75,
                    },
                },
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Fitted sphere wireframe.
        sphere_traces: list[go.Scatter3d] = []
        if show_fit:
            sphere_traces = self._sphere_wireframe(params)
            for tr in sphere_traces:
                fig.add_trace(tr)

        # Slider steps — each toggles one isosurface;
        # wall + colorbar + sphere wireframe always on.
        n_always_on = 2 + len(sphere_traces)  # wall, colorbar, sphere lines
        steps = []
        n_iso = len(iso_levels)
        for i, iso_val in enumerate(iso_levels):
            vis = [False] * n_iso + [True] * n_always_on
            vis[i] = True
            steps.append(
                {
                    "method": "update",
                    "args": [
                        {"visible": vis},
                        {
                            "title": title
                            or (f"Isosurface ρ = {iso_val:.4f} — {self.label}"),
                        },
                    ],
                    "label": f"{iso_val:.4f}",
                }
            )

        default_title = title or (f"Isosurface ρ = {iso_levels[0]:.4f} — {self.label}")
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "ρ = "},
                    "pad": {"t": 50},
                    "steps": steps,
                }
            ],
            title=default_title,
            template="plotly_white",
            scene={
                "xaxis_title": "ξ (Å)",
                "yaxis_title": "η (Å)",
                "zaxis_title": "z (Å)",
                "aspectmode": "data",
            },
        )
        if save_path:
            fig.write_html(save_path)
        return fig

    # ------------------------------------------------------------------
    # Internals — fitted-sphere wireframe for the 3D plot.
    # ------------------------------------------------------------------

    @staticmethod
    def _sphere_wireframe(
        params: dict,
        *,
        n_meridians: int = 12,
        n_parallels: int = 8,
        pts_per_line: int = 80,
    ) -> list[go.Scatter3d]:
        """Build wireframe traces for the fitted sphere above the wall.

        Only the portion of the sphere at or above the wall height
        ``zi_0`` is drawn; anything below is clipped.

        Parameters
        ----------
        params : dict
            Model parameters with keys ``R_eq``, ``xi_c``, ``yi_c``,
            ``zi_c``, ``zi_0``.
        n_meridians : int
            Number of longitude (meridian) great circles.
        n_parallels : int
            Number of latitude (parallel) circles evenly spaced
            from bottom to top of the sphere.
        pts_per_line : int
            Points per wireframe line segment.

        Returns
        -------
        list[go.Scatter3d]
            Wireframe line traces (meridians + parallels), clipped
            at the wall height.
        """
        R = float(params["R_eq"])
        xc = float(params.get("xi_c", 0.0))
        yc = float(params.get("yi_c", 0.0))
        zc = float(params["zi_c"])
        z0 = float(params["zi_0"])

        line_style: dict = {
            "color": "black",
            "width": 3,
            "dash": "dash",
        }
        traces: list[go.Scatter3d] = []

        def _add_clipped_trace(
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
        ) -> None:
            """Append line segments for the portion with ``z >= z0``.

            Contiguous runs of above-wall points are emitted as
            separate traces so that Plotly does not draw a line
            through the masked region.
            """
            mask = z >= z0
            if not mask.any():
                return
            # Find contiguous runs of True in *mask*.
            diff = np.diff(mask.astype(np.int8))
            starts = np.flatnonzero(diff == 1) + 1
            ends = np.flatnonzero(diff == -1) + 1
            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, len(mask)]
            for s, e in zip(starts, ends, strict=True):
                traces.append(
                    go.Scatter3d(
                        x=x[s:e],
                        y=y[s:e],
                        z=z[s:e],
                        mode="lines",
                        line=line_style,
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

        # Meridians (longitude great circles) — clipped at the wall.
        theta_arr = np.linspace(0.0, np.pi, pts_per_line)
        for k in range(n_meridians):
            phi = 2.0 * np.pi * k / n_meridians
            x = xc + R * np.sin(theta_arr) * np.cos(phi)
            y = yc + R * np.sin(theta_arr) * np.sin(phi)
            z = zc + R * np.cos(theta_arr)
            _add_clipped_trace(x, y, z)

        # Parallels (latitude circles) — skip rings below the wall.
        phi_arr = np.linspace(0.0, 2.0 * np.pi, pts_per_line)
        theta_parallels = np.linspace(0.0, np.pi, n_parallels + 2)[1:-1]
        for th in theta_parallels:
            z_ring = zc + R * np.cos(th)
            if z_ring < z0:
                continue
            r_ring = R * np.sin(th)
            x = xc + r_ring * np.cos(phi_arr)
            y = yc + r_ring * np.sin(phi_arr)
            z = np.full_like(phi_arr, z_ring)
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        return traces

    # ------------------------------------------------------------------
    # Internals — 3D source extraction.
    # ------------------------------------------------------------------

    def _extract_3d(
        self, source: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return ``(xi, yi, zi, density_3d, model_params)`` from a 3D source."""
        if isinstance(source, CoupledFit3DBatchResult):
            return (
                source.xi_grid,
                source.yi_grid,
                source.zi_grid,
                source.density,
                source.model_params,
            )
        if isinstance(source, CoupledFit3DResults):
            if not source.batches:
                raise ValueError("CoupledFit3DResults has no batches.")
            ref = source.batches[0]
            mean_density = np.stack([b.density for b in source.batches], axis=0).mean(
                axis=0
            )
            return (
                ref.xi_grid,
                ref.yi_grid,
                ref.zi_grid,
                mean_density,
                ref.model_params,
            )
        raise TypeError(
            f"plot_3d_isosurface requires a CoupledFit3D source, "
            f"got {type(source).__name__}."
        )

    # ------------------------------------------------------------------
    # Internals — source dispatch.
    # ------------------------------------------------------------------

    def _extract(
        self, source: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, str]:
        if isinstance(source, CoupledFit2DBatchResult):
            return (
                source.xi_grid,
                source.zi_grid,
                source.density,
                source.model_params,
                "",
            )
        if isinstance(source, CoupledFit2DResults):
            if not source.batches:
                raise ValueError("CoupledFit2DResults has no batches to plot.")
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
        if isinstance(source, CoupledFit3DBatchResult):
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
        if isinstance(source, CoupledFit3DResults):
            if not source.batches:
                raise ValueError("CoupledFit3DResults has no batches to plot.")
            ref3d: CoupledFit3DBatchResult = source.batches[0]
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
