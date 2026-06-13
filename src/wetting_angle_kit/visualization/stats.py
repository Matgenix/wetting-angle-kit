from dataclasses import dataclass


@dataclass
class TrajectoryStats:
    """Summary statistics for a single contact-angle trajectory.

    A plotter's ``.summary()`` method returns this dataclass so callers
    can both display the block (``print(stats)``) and reuse the
    underlying numbers programmatically.

    Attributes
    ----------
    method_name : str
        Name of the analysis method (e.g. ``"Slicing Analysis"``).
    label : str
        Display label identifying the trajectory.
    mean_surface_area : float
        Mean droplet/cap surface area in Å².
    mean_contact_angle : float
        Mean contact angle in degrees.
    std_contact_angle : float
        Standard deviation of the contact angle in degrees.
    n_samples : int
        Number of samples (frames or batches) contributing to the means.
    """

    method_name: str
    label: str
    mean_surface_area: float
    mean_contact_angle: float
    std_contact_angle: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"Label: {self.label}\n"
            f"Method: {self.method_name}\n"
            f"Mean Surface Area: {self.mean_surface_area:.4f}\n"
            f"Mean Contact Angle: {self.mean_contact_angle:.4f}°\n"
            f"Std Contact Angle: {self.std_contact_angle:.4f}°\n"
            f"N samples: {self.n_samples}"
        )
