"""Small shared grid helpers used across the analysis subpackages.

Kept dependency-free (numpy only) so both the grid interface extractor
and the coupled-fit analyzers can import it without a cross-subsystem
dependency.
"""

import numpy as np


def edges_from_bin_width(lo: float, hi: float, bin_width: float) -> np.ndarray:
    """Bin edges spanning ``[lo, hi]`` with cells of approximately ``bin_width``.

    The number of cells is rounded to the nearest integer; the range
    bounds are honoured exactly, so the effective cell width is
    ``(hi - lo) / n_cells`` which may differ slightly from
    ``bin_width``. Always returns at least one cell.
    """
    n = max(int(round((float(hi) - float(lo)) / float(bin_width))), 1)
    return np.linspace(float(lo), float(hi), n + 1)
