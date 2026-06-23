"""Unit tests for :class:`TemporalAggregator`."""

import pytest

from wetting_angle_kit.analysis.temporal import TemporalAggregator


def test_iter_batches_per_frame() -> None:
    agg = TemporalAggregator(batch_size=1)
    assert list(agg.iter_batches([0, 1, 2])) == [[0], [1], [2]]


def test_iter_batches_pooled() -> None:
    agg = TemporalAggregator(batch_size=2)
    assert list(agg.iter_batches([0, 1, 2, 3, 4])) == [[0, 1], [2, 3], [4]]


def test_iter_batches_fully_pooled() -> None:
    agg = TemporalAggregator(batch_size=-1)
    assert list(agg.iter_batches([0, 1, 2, 3])) == [[0, 1, 2, 3]]


def test_iter_batches_empty_returns_nothing() -> None:
    agg = TemporalAggregator(batch_size=1)
    assert list(agg.iter_batches([])) == []


@pytest.mark.parametrize("bad", [0, -2, -10])
def test_rejects_zero_or_invalid_negative(bad: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be"):
        TemporalAggregator(batch_size=bad)


@pytest.mark.parametrize(
    "n_frames,batch_size,expected",
    [
        (10, 1, 10),
        (10, 3, 4),
        (10, 10, 1),
        (10, -1, 1),
        (0, 1, 0),
        (0, -1, 0),
    ],
)
def test_n_batches(n_frames: int, batch_size: int, expected: int) -> None:
    assert TemporalAggregator(batch_size=batch_size).n_batches(n_frames) == expected
