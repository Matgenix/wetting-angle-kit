from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_surface_file(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return x,y columns from surface text file.

    Parameters
    ----------
    file_path : str
        Path to whitespace-delimited file with at least two columns.

    Returns
    -------
    tuple(ndarray, ndarray)
        (x, y) coordinate arrays.
    """
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def plot_slice(x: np.ndarray, y: np.ndarray) -> None:
    """Plot a 2D surface contour line from x and y coordinate arrays."""
    plt.figure()
    plt.plot(x, y, label="Surface Slice")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Slice of Fitted Surface")
    plt.legend()
    plt.grid()
    plt.show()


def read_surface_file(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load surface file returning x,y,z arrays (z zeros if absent).

    Parameters
    ----------
    file_path : str
        Path to surface file with 2 or 3 columns.

    Returns
    -------
    tuple(ndarray, ndarray, ndarray)
        (x, y, z) arrays; z is zeros if file has only two columns.
    """
    data = np.loadtxt(file_path)
    if data.shape[1] == 2:
        x, y = data[:, 0], data[:, 1]
        z = np.zeros_like(x)
    else:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
    return x, y, z


def plot_surface_and_points(
    x_surf: np.ndarray,
    y_surf: np.ndarray,
    z_surf: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
) -> None:
    """Render 3D plot of surface curve and point cloud.

    Parameters
    ----------
    x_surf, y_surf, z_surf : ndarray
        Surface coordinates.
    x_points, y_points, z_points : ndarray
        Point cloud coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_surf, y_surf, z_surf, label="Surface", color="black")
    ax.scatter(
        x_points, y_points, z_points, s=10, alpha=0.7, label="Points", color="tab:blue"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_surface_with_points(surface_file: str, points: np.ndarray) -> None:
    """Convenience wrapper: load surface and overlay points.

    Parameters
    ----------
    surface_file : str
        Path to surface file.
    points : ndarray, shape (N, 3)
        XYZ coordinates of points to overlay.
    """
    x_surf, y_surf, z_surf = read_surface_file(surface_file)
    x_points, y_points, z_points = points[:, 0], points[:, 1], points[:, 2]
    plot_surface_and_points(x_surf, y_surf, z_surf, x_points, y_points, z_points)


def plot_liquid_particles(
    positions: np.ndarray,
    ax: Any | None = None,
    color: str = "tab:blue",
    subsample: int | None = None,
) -> Any:
    """Scatter plot 3D particle positions with optional subsampling.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle coordinates.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing axes to plot on; new figure created if None.
    color : str, default "tab:blue"
        Marker color.
    subsample : int, optional
        If provided and smaller than N, random subset size to plot.

    Returns
    -------
    mpl_toolkits.mplot3d.Axes3D
        Axes object used for plotting.
    """
    if subsample is not None and subsample < len(positions):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(positions), size=subsample, replace=False)
        positions = positions[idx]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], s=8, alpha=0.8, color=color
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax
