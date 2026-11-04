"""Fixed anchors, window slots, record-start truncation, freshness
weights."""
from __future__ import annotations

import numpy as np
import pytest

from analog_ice import MonthCalendar, anchor_lag, slot_weights, \
    window_for_month
from analog_ice.windows import ANCHOR_LAGS


def test_two_months_per_lag() -> None:
    # exactly two calendar months per lag: Mar..Aug anchor at max with
    # d = 0..5, Sep..Feb anchor at min with d = 0..5
    expected = {0: [3, 9], 1: [4, 10], 2: [5, 11], 3: [6, 12], 4: [1, 7],
                5: [2, 8]}
    got: dict[int, list[int]] = {}
    for cal_month, (_, d) in ANCHOR_LAGS.items():
        got.setdefault(d, []).append(cal_month)
    assert len(ANCHOR_LAGS) == 12
    for d in range(6):
        assert sorted(got[d]) == expected[d]


def test_anchor_map() -> None:
    assert anchor_lag(3) == ("max", 0)    # March anchors at itself
    assert anchor_lag(9) == ("min", 0)    # September anchors at itself
    assert anchor_lag(8) == ("max", 5)    # August: 5 months after March
    assert anchor_lag(2) == ("min", 5)    # February: 5 months after Sep
    assert anchor_lag(1) == ("min", 4)    # January anchors previous Sep
    with pytest.raises(KeyError):
        anchor_lag(0)


def test_window_slots(calendar) -> None:
    # August 2001: window [March 2001, August 2001], lag 5
    idx = calendar.index_of(2001, 8)
    win = window_for_month(idx, calendar.months)
    assert win.lag == 5
    assert win.slots == tuple(range(calendar.index_of(2001, 3), idx + 1))
    assert win.anchor_type == "max-03"
    # January 2001 anchors September 2000 (lag 4)
    idx = calendar.index_of(2001, 1)
    win = window_for_month(idx, calendar.months)
    assert win.lag == 4
    assert win.slots[0] == calendar.index_of(2000, 9)
    assert win.anchor_type == "min-09"


def test_record_start_truncation() -> None:
    cal = MonthCalendar.make(1979, 60)
    win = window_for_month(0, cal.months)   # January 1979
    assert win.slots == (0,) and win.lag == 4
    win = window_for_month(1, cal.months)   # February 1979
    assert win.slots == (0, 1) and win.lag == 5
    win = window_for_month(2, cal.months)   # March 1979
    assert win.slots == (2,) and win.lag == 0


def test_slot_weights_full_profile() -> None:
    w = slot_weights(5, 6)
    assert np.allclose(w, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])


def test_slot_weights_tail_positions() -> None:
    # a 2-slot candidate of a depth-5 window keeps weights of slots 4, 5
    w = slot_weights(5, 2)
    assert np.allclose(w, [5 / 6, 1.0])
    # a single-slot candidate keeps only the freshest weight
    assert np.allclose(slot_weights(5, 1), [1.0])
    with pytest.raises(ValueError):
        slot_weights(5, 8)
