import os

import matplotlib.pyplot as plt
import numpy as np


class MethodComparison:
    """Utility to compare statistics from multiple trajectory analyzers.
    Parameters
    ----------
    analyzers : list
        Analyzer instances exposing ``directories`` and required API methods.
    method_names : list[str], optional
        Custom display names. If None, uses each analyzer's ``get_method_name``.
    """

    def __init__(self, analyzers, method_names=None):
        self.analyzers = analyzers
        self.method_names = method_names or [a.get_method_name() for a in analyzers]
        for analyzer in self.analyzers:
            if not hasattr(analyzer, "data") or not analyzer.data:
                analyzer.load_data()

    def _check_and_run_analysis(self, analyzer):
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
                    f"No analysis found for {directory}. "
                    "Please run the analysis first."
                )

    def _read_analysis_output(self, analyzer, directory):
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
        with open(output_file, "r") as f:
            lines = f.readlines()
            mean_surface_area = float(lines[2].split(": ")[1].strip())
            mean_contact_angle = float(lines[3].split(": ")[1].strip().replace("°", ""))
        return mean_surface_area, mean_contact_angle

    def plot_side_by_side_comparison(
        self, save_path=None, figsize=(14, 5), color="purple"
    ):
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

        for ax, analyzer, method_name in zip(axes, self.analyzers, self.method_names):
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
            xvals, yvals = np.array(xvals), np.array(yvals)
            if len(xvals) >= 2:
                coeffs = np.polyfit(xvals, yvals, 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(0, xvals.max() * 1.1, 100)
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

            if yvals.size > 0:
                ax.set_ylim(min(yvals) - 2, max(yvals) + 2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def plot_overlay_comparison(self, save_path=None, figsize=(8, 6), color="purple"):
        """
        Overlay mean angle vs surface area scaling across all analyzers.
        Inspired by plot_mean_angle_vs_surface().
        """
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 15,
                "legend.fontsize": 10,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 1.0,
                "errorbar.capsize": 3,
            }
        )

        fig, ax = plt.subplots(figsize=figsize)
        all_yvals = []

        for analyzer, method_name in zip(self.analyzers, self.method_names):
            xvals, yvals = [], []

            for directory in analyzer.directories:
                mean_sa, mean_angle = self._read_analysis_output(analyzer, directory)

                x = 1.0 / np.sqrt(mean_sa)
                y = mean_angle

                label = f"{method_name} – {analyzer.get_clean_label(directory)}"

                ax.errorbar(
                    x, y, yerr=0.5, fmt="o", color=color, alpha=0.7, label=label
                )

                xvals.append(x)
                yvals.append(y)

            # Fit per method
            xvals, yvals = np.array(xvals), np.array(yvals)
            if len(xvals) >= 2:
                coeffs = np.polyfit(xvals, yvals, 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(0, xvals.max() * 1.1, 100)
                ax.plot(
                    x_fit,
                    fit_line(x_fit),
                    "--",
                    label=f"{method_name} fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}",
                )

            all_yvals.extend(yvals)

        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title("Method Comparison: Mean Angle vs Surface Area")
        ax.legend(frameon=False, fontsize=7)
        ax.set_xlim(left=-0.001)

        if all_yvals:
            ax.set_ylim(min(all_yvals) - 2, max(all_yvals) + 2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def compare_statistics(self):
        """Print summary statistics aggregated across methods and directories."""
        print("=" * 70)
        print("METHOD COMPARISON STATISTICS")
        print("=" * 70)
        for method_name, analyzer in zip(self.method_names, self.analyzers):
            print(f"\n{method_name}:")
            print("-" * 70)
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
                print(f"  {analyzer.get_clean_label(directory)}:")
                print(f"    Mean Surface Area: {mean_surface_area:.4f}")
                print(f"    Mean Angle: {mean_contact_angle:.4f}°")
            if all_angles:
                print("\n  Overall Statistics:")
                print(f"    Total samples: {len(all_angles)}")
                print(f"    Mean Surface Area: {np.mean(all_surfaces):.4f}")
                print(f"    Mean Angle: {np.mean(all_angles):.4f}°")
                print(f"    Std Angle: {np.std(all_angles):.4f}°")
        print("\n" + "=" * 70)
