import os

import matplotlib.pyplot as plt
import numpy as np

from wetting_angle_kit.visualization.base_trajectory_analyzer import (
    BaseTrajectoryAnalyzer,
)


class SlicedTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """BaseTrajectoryAnalyzer implementation for the sliced contact angle method."""

    def __init__(
        self,
        directories: list[str],
        time_steps: dict[str, float] | None = None,
        time_unit: str = "ps",
    ) -> None:
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

    def _initialize_data_structure(self) -> None:
        """Initialize data structure for sliced analysis."""
        for directory in self.directories:
            self.data[directory] = {
                "surfaces_files": [],
                "popts_files": [],
                "angles_files": [],
                "mean_surface_areas": [],
                "all_angles": [],
                "median_angles": [],
                "mean_angles": [],
                "std_angles": [],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    def get_method_name(self) -> str:
        """Return method name."""
        return "Sliced Analysis"

    def load_data(self) -> None:
        """Load combined .npy files (angles, surfaces, popts)
        from all directories and compute mean surface areas per frame."""
        for directory in self.directories:
            all_angles = np.load(
                os.path.join(directory, "all_angles.npy"), allow_pickle=True
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
                "all_angles": all_angles,
                "all_surfaces": all_surfaces,
                "all_popts": all_popts,
                "frame_numbers": [item[0] for item in all_angles],
                "mean_surface_areas": mean_surface_areas,
                "median_angles": [(item[0], np.median(item[1])) for item in all_angles],
                "mean_angles": [(item[0], np.mean(item[1])) for item in all_angles],
                "std_angles": [(item[0], np.std(item[1])) for item in all_angles],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    @staticmethod
    def calculate_polygon_area(points: np.ndarray) -> float:
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

    def mean_surface_frame(self, surfaces: list[np.ndarray]) -> float:
        """Return the mean polygon area across all surfaces in a frame.

        Parameters
        ----------
        surfaces : list[ndarray]
            List of surface polygon vertex arrays, each of shape (N, 2).

        Returns
        -------
        float
            Mean surface area across all surfaces in the frame.
        """
        all_surf = [self.calculate_polygon_area(surface) for surface in surfaces]
        return np.mean(np.array(all_surf))

    def get_surface_areas(self, directory: str) -> np.ndarray:
        """Get surface areas for a directory."""
        return np.array(self.data[directory]["mean_surface_areas"])

    def get_contact_angles(self, directory: str) -> np.ndarray:
        """Get contact angles (median angles) for a directory."""
        data = np.array(self.data[directory]["median_angles"])
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1]
        return data

    def plot_median_angles_evolution(
        self,
        save_path: str,
        labels: dict[str, str] | None = None,
        plot_std: bool = True,
    ) -> None:
        """Plot evolution of median contact angle with standard deviation.

        Align trajectories by truncating to shortest.
        """
        if not self.data[self.directories[0]]["median_angles"]:
            self.load_data()

        # Use provided labels or fall back to directory basename
        plot_labels = (
            labels if labels else {d: os.path.basename(d) for d in self.directories}
        )

        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))  # type: ignore[attr-defined]

        for i, directory in enumerate(self.directories):
            median_angles = self.data[directory]["median_angles"]
            std_angles = self.data[directory]["std_angles"]
            frame_numbers = [item[0] for item in median_angles]
            median_values = [item[1] for item in median_angles]
            std_values = [item[1] for item in std_angles]
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

    def plot_mean_angles_evolution(
        self,
        save_path: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Plot evolution of mean contact angle with standard deviation."""
        plot_labels = (
            labels if labels else {d: os.path.basename(d) for d in self.directories}
        )
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))  # type: ignore[attr-defined]

        for i, directory in enumerate(self.directories):
            mean_angles = self.data[directory]["mean_angles"]
            std_angles = self.data[directory]["std_angles"]
            frame_numbers = [item[0] for item in mean_angles]
            mean_values = [item[1] for item in mean_angles]
            std_values = [item[1] for item in std_angles]
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

        plt.title("Evolution of the Mean Angle with Standard Deviation")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (°)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
