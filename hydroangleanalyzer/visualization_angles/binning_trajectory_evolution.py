import glob
import os
import re

import numpy as np

# Ensure this matches your actual import structure
from .base_trajectory_analyzer import BaseTrajectoryAnalyzer


class BinningTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """Analyze binning trajectory data using circular segment calculations."""

    def __init__(self, directories, split_factor=1, time_steps=None, time_unit="ps"):
        """
        Initialize the analyzer with a list of directory paths and split factor.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        split_factor : int, optional
            Number of batches/splits to process in each directory.
        time_steps : dict, optional
            Dictionary mapping directory to its time step.
        time_unit : str, optional
            Time unit for the x-axis (e.g., "ps", "ns", "fs").
        """
        self.split_factor = split_factor
        self.time_steps = time_steps if time_steps else {d: 1.0 for d in directories}

        # Initialize Base Class (this will trigger _initialize_data_structure)
        super().__init__(directories, time_unit=time_unit)

    def _initialize_data_structure(self):
        """Initialize data structure for binning analysis."""
        for directory in self.directories:
            self.data[directory] = {
                "R_eq": [],
                "zi_c": [],
                "zi_0": [],
                "contact_angles": [],
                "surface_areas": [],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    def get_method_name(self):
        """Return method name."""
        return "Binning Analysis"

    @staticmethod
    def circular_segment_area(R, z_center, z_cut):
        """Compute the area of a circular segment for any cut position."""
        h = (z_center + R) - z_cut  # height of the cap
        if h <= 0:
            return 0.0
        if h >= 2 * R:
            return np.pi * R**2
        if h <= R:
            return R**2 * np.arccos((R - h) / R) - (R - h) * np.sqrt(2 * R * h - h**2)
        else:
            h_small = 2 * R - h
            return np.pi * R**2 - (
                R**2 * np.arccos((R - h_small) / R)
                - (R - h_small) * np.sqrt(2 * R * h_small - h_small**2)
            )

    def load_files(self):
        """Load and sort all relevant log files from each directory."""
        for directory in self.directories:
            log_files = sorted(
                glob.glob(os.path.join(directory, "log_data_batch_*.txt")),
                key=lambda x: int(re.search(r"batch_(\d+)", x).group(1)),
            )
            if not log_files:
                raise ValueError(
                    f"No log_data_batch_*.txt files found in directory: {directory}"
                )
            self.data[directory]["log_files"] = log_files

    def read_data(self):
        """Alias for load_data for backward compatibility."""
        self.load_data()

    def load_data(self):
        """Read and parse data from log files in each directory."""
        self.load_files()
        for directory in self.directories:
            # Clear previous data for this directory
            self.data[directory]["R_eq"] = []
            self.data[directory]["zi_c"] = []
            self.data[directory]["zi_0"] = []
            self.data[directory]["contact_angles"] = []
            self.data[directory]["surface_areas"] = []
            print(self.data[directory]["log_files"])
            # Read all batch log files for this directory
            for log_file in self.data[directory]["log_files"]:
                with open(log_file, "r") as f:
                    text = f.read()

                # Extract R_eq
                R_eq_match = re.search(r"R_eq:([0-9\.\-eE]+)", text)
                if not R_eq_match:
                    raise ValueError(f"R_eq not found in file: {log_file}")
                R_eq = float(R_eq_match.group(1))

                # Extract zi_c
                zi_c_match = re.search(r"zi_c:([0-9\.\-eE]+)", text)
                if not zi_c_match:
                    raise ValueError(f"zi_c not found in file: {log_file}")
                zi_c = float(zi_c_match.group(1))

                # Extract zi_0
                zi_0_match = re.search(r"zi_0:([0-9\.\-eE]+)", text)
                if not zi_0_match:
                    raise ValueError(f"zi_0 not found in file: {log_file}")
                zi_0 = float(zi_0_match.group(1))

                # Extract contact angle
                angle_match = re.search(r"Contact angle:([0-9\.\-eE]+)", text)
                if not angle_match:
                    raise ValueError(f"Contact angle not found in file: {log_file}")
                angle = float(angle_match.group(1))

                # Calculate surface area
                A_seg = self.circular_segment_area(R_eq, zi_c, zi_0)

                # Append data
                self.data[directory]["R_eq"].append(R_eq)
                self.data[directory]["zi_c"].append(zi_c)
                self.data[directory]["zi_0"].append(zi_0)
                self.data[directory]["contact_angles"].append(angle)
                self.data[directory]["surface_areas"].append(A_seg)

    def get_surface_areas(self, directory):
        """Return surface areas for a directory."""
        return np.array(self.data[directory]["surface_areas"])

    def get_contact_angles(self, directory):
        """Return contact angles for a directory."""
        return np.array(self.data[directory]["contact_angles"])
