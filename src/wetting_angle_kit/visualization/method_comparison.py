import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class MethodComparison:
    """Utility to compare contact angle statistics
    from multiple trajectory analyzers."""

    def __init__(
        self, analyzers: list[Any], method_names: list[str] | None = None
    ) -> None:
        """
        Parameters
        ----------
        analyzers : list
            Analyzer instances exposing ``directories`` and required API methods.
        method_names : list[str], optional
            Custom display names. If None, uses each analyzer's ``get_method_name``.
        """
        self.analyzers = analyzers
        self.method_names = method_names or [a.get_method_name() for a in analyzers]
        for analyzer in self.analyzers:
            if not hasattr(analyzer, "data") or not analyzer.data:
                analyzer.load_data()

    def _check_and_run_analysis(self, analyzer: Any) -> None:
        """Run analyzer if expected output file is absent for any directory.
        Parameters
        ----------
        analyzer : BaseTrajectoryAnalyzer
            Analyzer instance whose output will be checked.
        """
        for directory in analyzer.directories:
            output_file = f"{directory}/output_stats.txt"
            if not os.path.exists(output_file):
                raise FileNotFoundError(
                    f"No analysis found for {directory}. Please run the analysis first."
                )

    def _read_analysis_output(
        self, analyzer: Any, directory: str
    ) -> tuple[float, float]:
        """Return mean surface area and angle parsed from stats file.
        Parameters
        ----------
        analyzer : BaseTrajectoryAnalyzer
            Analyzer owning the directory.
        directory : str
            Path containing ``output_stats.txt``.
        Returns
        -------
        tuple(float, float)
            (mean_surface_area, mean_contact_angle).
        """
        output_file = f"{directory}/output_stats.txt"
        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()
            mean_surface_area = float(lines[2].split(": ")[1].strip())
            mean_contact_angle = float(lines[3].split(": ")[1].strip().replace("°", ""))
        return mean_surface_area, mean_contact_angle

    def plot_side_by_side_comparison(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (14, 5),
        color: str = "purple",
    ) -> None:
        """
        Produce side-by-side comparison of mean contact angle vs. surface area scaling.
        Inspired by plot_mean_angle_vs_surface().
        """
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 15,
                "legend.fontsize": 11,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 1.0,
                "errorbar.capsize": 3,
            }
        )
        fig, axes = plt.subplots(1, len(self.analyzers), figsize=figsize)
        if len(self.analyzers) == 1:
            axes = [axes]

        for ax, analyzer, method_name in zip(
            axes, self.analyzers, self.method_names, strict=False
        ):
            # gather one point per directory
            xvals, yvals = [], []
            for directory in analyzer.directories:
                mean_sa, mean_angle = self._read_analysis_output(analyzer, directory)

                x = 1.0 / np.sqrt(mean_sa)  # same as example
                y = mean_angle

                ax.errorbar(x, y, yerr=0.5, fmt="o", color=color)
                ax.annotate(
                    analyzer.get_clean_label(directory),
                    xy=(x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

                xvals.append(x)
                yvals.append(y)

            # linear fit if we have ≥2 points
            xvals_arr, yvals_arr = np.array(xvals), np.array(yvals)
            if len(xvals_arr) >= 2:
                coeffs = np.polyfit(xvals_arr, yvals_arr, 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(0, xvals_arr.max() * 1.1, 100)
                ax.plot(
                    x_fit,
                    fit_line(x_fit),
                    "--",
                    color="gray",
                    label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}",
                )

            ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
            ax.set_ylabel("Mean Angle (°)")
            ax.set_title(method_name)
            ax.legend(frameon=False)
            ax.set_xlim(left=-0.001)

            if yvals_arr.size > 0:
                ax.set_ylim(min(yvals_arr) - 2, max(yvals_arr) + 2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def plot_overlay_comparison(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (7, 5),
        colors: list[str] | None = None,
    ) -> None:
        """Overlay Modified Young's equation plot across all analyzers.

        Plots ``cos(θ)`` vs ``1/√A`` with linear fits and extrapolated
        ``θ∞`` (contact angle at infinite surface area) for each analyzer.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        figsize : tuple[int, int], default (7, 5)
            Figure size in inches.
        colors : list[str], optional
            One color per analyzer. If None a default palette is used.
        """
        if colors is None:
            colors = ["#0A9396", "#bb3e03", "#9b5de5", "#f15bb5", "#00bbf9"]

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 15,
                "legend.fontsize": 11,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 1.0,
                "errorbar.capsize": 3,
            }
        )

        fig, ax = plt.subplots(figsize=figsize)

        for idx, (analyzer, method_name) in enumerate(
            zip(self.analyzers, self.method_names, strict=False)
        ):
            color = colors[idx % len(colors)]
            xvals, yvals = [], []

            for directory in analyzer.directories:
                mean_sa, mean_angle = self._read_analysis_output(analyzer, directory)
                # Read std from output_stats.txt (line 4)
                output_file = f"{directory}/output_stats.txt"
                with open(output_file, encoding="utf-8") as f:
                    lines = f.readlines()
                std_angle = float(lines[4].split(": ")[1].strip().replace("°", ""))

                x = 1.0 / np.sqrt(mean_sa)
                y = np.cos(np.radians(mean_angle))

                # Error propagation: d(cos θ) = |sin θ| · dθ
                yerr = (
                    np.abs(np.sin(np.radians(mean_angle))) * np.radians(std_angle) / 5
                )

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="o",
                    color=color,
                    markersize=6,
                    capsize=3,
                    lw=1.2,
                )
                ax.annotate(
                    analyzer.get_clean_label(directory),
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                )

                xvals.append(x)
                yvals.append(y)

            # Linear fit with θ∞ extrapolation
            if len(xvals) >= 2:
                xarr, yarr = np.array(xvals), np.array(yvals)
                coeffs = np.polyfit(xarr, yarr, 1)
                fit_line = np.poly1d(coeffs)
                intercept = np.clip(coeffs[1], -1.0, 1.0)
                theta_inf = np.degrees(np.arccos(intercept))

                x_fit = np.linspace(0, xarr.max() * 1.1, 100)
                ax.plot(
                    x_fit,
                    fit_line(x_fit),
                    "--",
                    color=color,
                    lw=1.5,
                    label=(
                        f"{method_name}: "
                        rf"$\theta_{{\infty}} = {theta_inf:.1f}^\circ$"
                    ),
                )

        ax.set_xlabel(r"$1 / \sqrt{A} \; (\mathrm{\AA^{-1}})$")
        ax.set_ylabel(r"$\cos(\theta)$")
        ax.set_title("Modified Young's Eq – Comparison")
        ax.legend(frameon=False, loc="best")
        ax.set_xlim(left=-0.001)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def compare_statistics(self) -> str:
        """Build per-directory and overall mean/std statistics for each method.

        Returns
        -------
        str
            The full formatted report. The caller is responsible for displaying
            it (e.g. ``print(comparator.compare_statistics())``). Returning a
            string instead of printing makes the output capturable from
            notebooks and tests.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("METHOD COMPARISON STATISTICS")
        lines.append("=" * 70)
        for method_name, analyzer in zip(
            self.method_names, self.analyzers, strict=False
        ):
            lines.append(f"\n{method_name}:")
            lines.append("-" * 70)
            all_angles = []
            all_surfaces = []
            for directory in analyzer.directories:
                try:
                    mean_surface_area, mean_contact_angle = self._read_analysis_output(
                        analyzer, directory
                    )
                    angles = analyzer.get_contact_angles(directory)
                    surfaces = analyzer.get_surface_areas(directory)
                except FileNotFoundError:
                    angles = analyzer.get_contact_angles(directory)
                    surfaces = analyzer.get_surface_areas(directory)
                    mean_surface_area = float(np.mean(surfaces))
                    mean_contact_angle = float(np.mean(angles))
                all_angles.extend(angles)
                all_surfaces.extend(surfaces)
                lines.append(f"  {analyzer.get_clean_label(directory)}:")
                lines.append(f"    Mean Surface Area: {mean_surface_area:.4f}")
                lines.append(f"    Mean Angle: {mean_contact_angle:.4f}°")
            if all_angles:
                lines.append("\n  Overall Statistics:")
                lines.append(f"    Total samples: {len(all_angles)}")
                lines.append(f"    Mean Surface Area: {np.mean(all_surfaces):.4f}")
                lines.append(f"    Mean Angle: {np.mean(all_angles):.4f}°")
                lines.append(f"    Std Angle: {np.std(all_angles):.4f}°")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
