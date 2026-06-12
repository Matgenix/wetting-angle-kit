"""Exercise the ``_run_parallel`` (``multiprocessing.Pool``) path.

The fast-path optimisation makes single-batch and ``n_jobs=1`` calls
go inline, so the parallel branch needs an explicit kick to cover it.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("ovito")

from wetting_angle_kit.analysis import CoupledFit2DAnalyzer  # noqa: E402
from wetting_angle_kit.analysis.temporal import TemporalAggregator  # noqa: E402
from wetting_angle_kit.parsers import (  # noqa: E402
    LammpsDumpParser,
    LammpsDumpWaterFinder,
)

_FIXTURE = (
    pathlib.Path(__file__).parent
    / ".."
    / "trajectories"
    / "traj_spherical_drop_4k.lammpstrj"
)


@pytest.mark.integration
def test_run_parallel_path_executes_with_n_jobs_2() -> None:
    """Multi-batch + ``n_jobs=2`` forces the ``_run_parallel`` branch.

    Coupled binning is the cheapest analyzer; using it here keeps the
    test under ~5 seconds even with two real worker processes.
    """
    oxygen_indices = LammpsDumpWaterFinder(
        _FIXTURE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params={
            "xi_0": 0.0,
            "xi_f": 40.0,
            "bin_width_x": 1.4,
            "zi_0": 0.0,
            "zi_f": 40.0,
            "bin_width_z": 1.4,
        },
        temporal_aggregator=TemporalAggregator(batch_size=1),
    )
    # Multi-batch (len > 1) AND n_jobs > 1 → parallel path.
    results = analyzer.analyze([0, 1, 2], n_jobs=2)
    assert len(results) == 3
    for batch in results.batches:
        assert np.isfinite(batch.angle)


@pytest.mark.integration
def test_n_jobs_gt_1_with_batch_size_minus_1_warns_and_runs_inline() -> None:
    """``batch_size=-1`` + ``n_jobs > 1`` warns: no parallelism possible.

    With ``batch_size=-1`` every frame is pooled into one batch, so
    ``n_jobs`` can't subdivide the work. The analyzer falls back to
    inline execution and emits a ``UserWarning`` to flag the
    mis-configuration.
    """
    oxygen_indices = LammpsDumpWaterFinder(
        _FIXTURE, oxygen_type=1, hydrogen_type=2
    ).get_water_oxygen_ids(0)
    analyzer = CoupledFit2DAnalyzer(
        parser=LammpsDumpParser(_FIXTURE),
        atom_indices=oxygen_indices,
        droplet_geometry="spherical",
        binning_params={
            "xi_0": 0.0,
            "xi_f": 40.0,
            "bin_width_x": 1.4,
            "zi_0": 0.0,
            "zi_f": 40.0,
            "bin_width_z": 1.4,
        },
        temporal_aggregator=TemporalAggregator(batch_size=-1),
    )
    with pytest.warns(UserWarning, match="batch_size=-1"):
        results = analyzer.analyze([0, 1, 2], n_jobs=4)
    # One pooled batch ⇒ one result regardless of n_jobs request.
    assert len(results) == 1
    assert np.isfinite(results.batches[0].angle)
