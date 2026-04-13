import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class BaseTrajectoryAnalyzer(ABC):
    def __init__(self, directories, time_unit="ps"):
        """
        Initialize the analyzer with a list of directory paths.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        time_unit : str, optional
            Time unit for the x-axis (e.g., "ps", "ns", "fs").
        """
        self.directories = directories
        self.data = {}
        self.time_unit = time_unit
        self._initialize_data_structure()

    @abstractmethod
    def _initialize_data_structure(self):
        """Initialize the data dictionary structure for each directory."""
        pass

    @abstractmethod
    def load_data(self):
        """Read and parse data from files in each directory."""
        pass

    @abstractmethod
    def get_surface_areas(self, directory):
        """
        Get surface areas for a given directory.

        Parameters
        ----------
        directory : str
            Directory path.

        Returns
        -------
        numpy.ndarray
            Array of surface area values.
        """
        pass

    @abstractmethod
    def get_contact_angles(self, directory):
        """
        Get contact angles for a given directory.

        Parameters
        ----------
        directory : str
            Directory path.

        Returns
        -------
        numpy.ndarray
            Array of contact angle values.
        """
        pass

    @abstractmethod
    def get_method_name(self):
        """
        Return the name of this analysis method.

        Returns
        -------
        str
            Method name for labels and titles.
        """
        pass

    def compute_statistics(self, directory):
        """
        Compute mean surface area, mean angle, and standard error.

        Parameters
        ----------
        directory : str
            Directory path.

        Returns
        -------
        tuple
            (x_value, y_value, y_error) where:
            - x_value: 1/sqrt(mean_surface_area)
            - y_value: mean contact angle
            - y_error: standard error of the mean
        """
        surface_areas = self.get_surface_areas(directory)
        contact_angles = self.get_contact_angles(directory)

        x = 1 / np.sqrt(np.mean(surface_areas))
        y = np.mean(contact_angles)
        yerr = np.std(contact_angles) / np.sqrt(len(contact_angles))

        return x, y, yerr

    def get_clean_label(self, directory):
        """
        Generate a clean label from directory name.

        Parameters
        ----------
        directory : str
            Directory path.

        Returns
        -------
        str
            Cleaned directory name for plotting.
        """
        return (
            directory.replace("_reduce_sliced", "")
            .replace("_reduce_binning", "")
            .replace("result_dump_", "")
        )

    def analyze(self, output_filename="output_stats.txt"):
        """Analyze and save statistics for each directory to an output file."""
        self.load_data()
        for directory in self.directories:
            output_path = f"{directory}/{output_filename}"
            with open(output_path, "w") as f:
                f.write(f"Directory: {directory}\n")
                f.write(f"Method: {self.get_method_name()}\n")
                f.write(
                    "Mean Surface Area: "
                    f"{np.mean(self.get_surface_areas(directory)):.4f}\n"
                )
                f.write(
                    "Mean Contact Angle: "
                    f"{np.mean(self.get_contact_angles(directory)):.4f}\u00b0\n"
                )
                f.write(
                    "Std Contact Angle: "
                    f"{np.std(self.get_contact_angles(directory)):.4f}\u00b0\n"
                )
            print(f"Analysis saved to: {output_path}")

    def plot_mean_angle_vs_surface(self, labels=None, color=None, save_path=None):
        """
        Generate a plot comparing mean angle vs surface
        area scaling. If no analysis output is found, run the analysis first.

        Parameters
        ----------
        labels : list of str, optional
            Labels for each dataset. If None, directory names are used.
        color : str, optional
            Base color for all datasets. If None, a default
            set of unique colors is used.
        save_path : str, optional
            Path to save the figure.
        """
        # Check if analysis output files exist; if not, run analysis
        for directory in self.directories:
            output_file = f"{directory}/output_stats.txt"
            if not os.path.exists(output_file):
                print(f"No analysis found for {directory}. Running analysis...")
                self.analyze()  # Run analysis to generate output files
                break  # Only need to run once

        # Read data if not already loaded
        if not hasattr(self, "data") or not self.data:
            self.load_data()

        # Set up plot parameters
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 15,
                "legend.fontsize": 12,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 1.0,
                "errorbar.capsize": 3,
            }
        )

        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Set default labels and colors if not provided
        if labels is None:
            labels = [self.get_clean_label(d) for d in self.directories]

        if color is None:
            color = "purple"
        colors = [color] * len(self.directories)
        # Collect data for plotting
        xvals, yvals = [], []
        for d, label, color in zip(self.directories, labels, colors):
            # Read data from the analysis output file
            output_file = f"{d}/output_stats.txt"
            with open(output_file, "r") as f:
                lines = f.readlines()
                mean_surface_area = float(lines[2].split(": ")[1].strip())
                mean_contact_angle = float(
                    lines[3].split(": ")[1].strip().replace("°", "")
                )
                std_contact_angle = float(
                    lines[4].split(": ")[1].strip().replace("°", "")
                )

            # Use the data for plotting
            x = 1 / np.sqrt(mean_surface_area)
            # Convert angle to radians for cosine calculation
            mean_angle_rad = np.radians(mean_contact_angle)
            y = np.cos(mean_angle_rad)

            # Propagate error: d(cos(theta)) = |-sin(theta)| * d(theta)
            # d(theta) must be in radians
            std_angle_rad = np.radians(std_contact_angle)
            yerr = np.abs(np.sin(mean_angle_rad)) * (std_angle_rad / 5)

            ax.errorbar(
                x, y, yerr=yerr, fmt="o", color=color, markersize=6, capsize=3, lw=1.2
            )
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6,
                color="black",
            )
            xvals.append(x)
            yvals.append(y)

        # Linear fit
        if len(xvals) >= 2:
            xvals, yvals = np.array(xvals), np.array(yvals)
            coeffs = np.polyfit(xvals, yvals, 1)
            fit_line = np.poly1d(coeffs)

            # Calculate theta_infinity from intercept (x=0)
            # intercept = cos(theta_inf) -> theta_inf = arccos(intercept)
            intercept = coeffs[1]
            # Clip to valid domain [-1, 1] just in case fit goes wild,
            # though unlikely for physical data
            intercept_clipped = np.clip(intercept, -1.0, 1.0)
            theta_inf_deg = np.degrees(np.arccos(intercept_clipped))

            x_fit = np.linspace(0, max(xvals) * 1.1, 100)
            ax.plot(
                x_fit,
                fit_line(x_fit),
                "--",
                color="gray",
                lw=1.5,
                label=(
                    f"Fit: $\\cos(\\theta) = {coeffs[0]:.2f}x + {coeffs[1]:.2f}$\n"
                    f"$\\theta_\\infty = {theta_inf_deg:.1f}^\\circ$"
                ),
            )

        # Set plot labels and title
        ax.set_xlabel(r"$1 / \sqrt{A} \; (\mathrm{\AA^{-1}})$")
        ax.set_ylabel(r"$\cos(\theta)$")
        ax.set_title(f"{self.get_method_name()} - Modified Young's Eq Plot", pad=10)
        ax.legend(frameon=False, loc="best")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        # Adjust ylim for cosine values (typically -1 to 1, but zoom in on data)
        if len(yvals) > 0:
            margin = (max(yvals) - min(yvals)) * 0.2 if len(yvals) > 1 else 0.1
            if margin == 0:
                margin = 0.1
            ax.set_ylim(bottom=min(yvals) - margin, top=max(yvals) + margin)
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
