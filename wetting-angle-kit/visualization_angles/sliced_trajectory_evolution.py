import os

import matplotlib.pyplot as plt
import numpy as np

from .base_trajectory_analyzer import BaseTrajectoryAnalyzer


class SlicedTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    def __init__(self, directories, time_steps=None, time_unit="ps"):
        """
        Initialize the analyzer with a list of directory paths.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        time_steps : dict, optional
            Dictionary mapping directory to its time step.
            If None, defaults to 1.0 for all directories.
        time_unit : str, optional
            Time unit for the x-axis (e.g., "ps", "ns", "fs").
        """
        self.time_steps = time_steps if time_steps else {d: 1.0 for d in directories}
        self.time_unit = time_unit
        super().__init__(directories, time_unit=time_unit)
        for directory in directories:
            self.data[directory] = {}

    def _initialize_data_structure(self):
        """Initialize data structure for sliced analysis."""
        for directory in self.directories:
            self.data[directory] = {
                "surfaces_files": [],
                "popts_files": [],
                "alfas_files": [],
                "mean_surface_areas": [],
                "all_alfas": [],
                "median_alfas": [],
                "mean_alfas": [],
                "std_alfas": [],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    def get_method_name(self):
        """Return method name."""
        return "Sliced Analysis"

    def load_data(self):
        """Load the combined .npy files for all
        directories and calculate mean surface areas per frame."""
        for directory in self.directories:
            all_alfas = np.load(
                os.path.join(directory, "all_alfas.npy"), allow_pickle=True
            )
            all_surfaces = np.load(
                os.path.join(directory, "all_surfaces.npy"), allow_pickle=True
            )
            all_popts = np.load(
                os.path.join(directory, "all_popts.npy"), allow_pickle=True
            )

            # Calculate mean surface area for each frame
            mean_surface_areas = []
            for frame_data in all_surfaces:
                surfaces = frame_data[1]
                all_surf = [
                    self.calculate_polygon_area(surface) for surface in surfaces
                ]
                mean_area = np.mean(np.array(all_surf))
                mean_surface_areas.append(mean_area)

            self.data[directory] = {
                "all_alfas": all_alfas,
                "all_surfaces": all_surfaces,
                "all_popts": all_popts,
                "frame_numbers": [item[0] for item in all_alfas],
                "mean_surface_areas": mean_surface_areas,
                "median_alfas": [(item[0], np.median(item[1])) for item in all_alfas],
                "mean_alfas": [(item[0], np.mean(item[1])) for item in all_alfas],
                "std_alfas": [(item[0], np.std(item[1])) for item in all_alfas],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    @staticmethod
    def calculate_polygon_area(points):
        """
        Calculate the area of a polygon given its vertices using the Shoelace formula.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, 2) containing polygon vertices.

        Returns
        -------
        float
            Area of the polygon.
        """
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def mean_surface_frame(self, surfaces):
        """
        Calculate the mean surface area for a given surfaces file.

        Parameters
        ----------
        surfaces_file : str
            Path to the surfaces file.

        Returns
        -------
        float
            Mean surface area across all surfaces in the frame.
        """
        all_surf = [self.calculate_polygon_area(surface) for surface in surfaces]
        return np.mean(np.array(all_surf))

    def get_surface_areas(self, directory):
        """Get surface areas for a directory."""
        return np.array(self.data[directory]["mean_surface_areas"])

    def get_contact_angles(self, directory):
        """Get contact angles (median alfas) for a directory."""
        data = np.array(self.data[directory]["median_alfas"])
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1]
        return data

    def plot_median_alfas_evolution(self, save_path, labels=None, plot_std=True):
        """Plot evolution of median angle (Alfas) with standard deviation.

        Align trajectories by truncating to shortest.
        """
        if not self.data[self.directories[0]]["median_alfas"]:
            self.analyze_alfas_only()

        # Use provided labels or fall back to directory basename
        plot_labels = (
            labels if labels else {d: os.path.basename(d) for d in self.directories}
        )

        # Find the minimum number of frames across all directories
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            median_alfas = self.data[directory]["median_alfas"]
            std_alfas = self.data[directory]["std_alfas"]
            frame_numbers = [item[0] for item in median_alfas]
            median_values = [item[1] for item in median_alfas]
            std_values = [item[1] for item in std_alfas]
            time_step = self.data[directory]["time_step"]
            time_values = np.array(frame_numbers) * time_step
            label = plot_labels.get(directory, os.path.basename(directory))

            plt.plot(
                time_values,
                median_values,
                linestyle="-",
                color=colors[i],
                label=f"{label}",
            )
            if plot_std:
                plt.fill_between(
                    time_values,
                    np.array(median_values) - np.array(std_values),
                    np.array(median_values) + np.array(std_values),
                    color=colors[i],
                    alpha=0.2,
                )

        plt.title("Evolution of the Median Angle")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (°)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def plot_mean_alfas_evolution(self, save_path, labels=None):
        """Plot evolution of mean angle (Alfas) with standard deviation."""
        plot_labels = (
            labels if labels else {d: os.path.basename(d) for d in self.directories}
        )
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            mean_alfas = self.data[directory]["mean_alfas"]
            std_alfas = self.data[directory]["std_alfas"]
            frame_numbers = [item[0] for item in mean_alfas]
            mean_values = [item[1] for item in mean_alfas]
            std_values = [item[1] for item in std_alfas]
            time_step = self.data[directory]["time_step"]
            time_values = np.array(frame_numbers) * time_step
            label = plot_labels.get(directory, os.path.basename(directory))

            plt.plot(
                time_values, mean_values, linestyle="-", color=colors[i], label=label
            )
            plt.fill_between(
                time_values,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                color=colors[i],
                alpha=0.2,
            )

        plt.title("Evolution of the Mean Angle (Alfas) with Standard Deviation")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (°)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
