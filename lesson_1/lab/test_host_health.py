import math

import pytest

from host_health import deviation_score

GRID = [i / 10 for i in range(11)]


@pytest.mark.parametrize("cpu, mem, expected", [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.6, 0.8, 1.0),
    (1.0, 1.0, math.sqrt(2)),
    (0.6, 0.4, 0.72111),
])
def test_known_points(cpu, mem, expected):
    assert deviation_score(cpu, mem) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("cpu, mem", [
    (-0.01, 0.5), (1.01, 0.5), (0.5, -0.01), (0.5, 1.01),
    (1.25, 0.5), (-0.2, 0.6), (0.5, 1.25), (1.5, 1.5), (-0.1, -0.1),
])
def test_out_of_range_raises(cpu, mem):
    with pytest.raises(ValueError):
        deviation_score(cpu, mem)


@pytest.mark.parametrize("cpu, mem, expected", [
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (1.0, 0.5, math.hypot(1.0, 0.5)),
    (0.5, 1.0, math.hypot(0.5, 1.0)),
])
def test_on_the_boundary_lines(cpu, mem, expected):
    assert deviation_score(cpu, mem) == pytest.approx(expected)


def test_boundaries_accepted():
    assert deviation_score(0.0, 0.0) == 0.0
    assert deviation_score(1.0, 1.0) == pytest.approx(math.sqrt(2))


def test_range_holds_across_grid():
    for a in GRID:
        for b in GRID:
            assert 0.0 <= deviation_score(a, b) <= math.sqrt(2) + 1e-12


def test_symmetric_across_grid():
    for a in GRID:
        for b in GRID:
            assert deviation_score(a, b) == pytest.approx(deviation_score(b, a))


def test_monotonic_in_each_axis():
    for b in GRID:
        row = [deviation_score(a, b) for a in GRID]
        assert row == sorted(row)
    for a in GRID:
        col = [deviation_score(a, b) for b in GRID]
        assert col == sorted(col)


def test_single_axis_reduces_to_input():
    for a in GRID:
        assert deviation_score(a, 0.0) == pytest.approx(a)
        assert deviation_score(0.0, a) == pytest.approx(a)
