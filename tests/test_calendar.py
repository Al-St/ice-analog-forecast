"""Calendar grid: round trips, labels, boundary errors."""
from __future__ import annotations

import pytest

from analog_ice import MonthCalendar


def test_make_and_labels(calendar) -> None:
    assert calendar.n_months == 540
    assert calendar.months[0] == 1 and calendar.years[0] == 1979
    assert calendar.months[11] == 12 and calendar.years[11] == 1979
    assert calendar.months[12] == 1 and calendar.years[12] == 1980
    assert calendar.label(0) == "1979-01"
    assert calendar.label(539) == "2023-12"


def test_index_round_trip(calendar) -> None:
    for idx in (0, 1, 11, 12, 300, 539):
        y, m = calendar.year_month(idx)
        assert calendar.index_of(y, m) == idx


def test_index_errors(calendar) -> None:
    with pytest.raises(ValueError):
        calendar.index_of(1979, 0)
    with pytest.raises(ValueError):
        calendar.index_of(1979, 13)
    with pytest.raises(ValueError):
        calendar.index_of(1968, 5)  # before the record
    with pytest.raises(ValueError):
        calendar.index_of(2024, 1)  # after the record
