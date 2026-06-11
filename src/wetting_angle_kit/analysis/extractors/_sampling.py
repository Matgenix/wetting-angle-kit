"""Direction-generation helpers used by the ray-based extractors."""

import numpy as np


def _fibonacci_sphere_directions(n: int) -> np.ndarray:
    """Equal-area Fibonacci-spiral directions on the full sphere.

    ``cos θ`` is uniformly spaced over ``[-1, 1]`` (so the surface
    density is uniform over the whole sphere) and ``φ`` is incremented
    by the golden angle for low-discrepancy azimuthal coverage.
    ``i = 0`` sits at the south pole (``cos θ = -1``) and
    ``i = n - 1`` at the north pole (``cos θ = 1``).

    The full sphere coverage is important for sessile droplets: rays
    emitted from the droplet COM in downward directions traverse the
    liquid, hit the wall plane, and contribute interface points at the
    wall — making :meth:`WallDetector.min_plus_offset` work correctly
    in the whole-fit pipeline. (Restricting to the upper hemisphere
    misses the wall, so ``min(shell z)`` lands on ``COM_z`` instead.)

    Parameters
    ----------
    n : int
        Number of directions.

    Returns
    -------
    ndarray, shape (n, 3)
        Unit direction vectors covering the full sphere.
    """
    if n <= 0:
        return np.empty((0, 3))
    i = np.arange(n, dtype=np.float64)
    cos_theta = 2.0 * i / (n - 1) - 1.0 if n > 1 else np.array([1.0])
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = (i * golden_angle) % (2.0 * np.pi)
    return np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )
