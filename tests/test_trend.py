"""Cell-wise OLS trend slopes: exactness on ramps, NaN guard, causality,
minimum-points rule."""
from __future__ import annotations

import numpy as np
import pytest

from analog_ice import MonthCalendar, ols_slopes, trend_slopes


def test_ols_slopes_exact_on_ramp() -> None:
    values = np.stack([np.full((3, 2), 5.0), np.full((3, 2), 8.0)],
                      axis=-1)  # +3 per occurrence
    slopes, n_bad = ols_slopes(values)
    assert np.allclose(slopes, 3.0)
    assert n_bad == 0


def test_ols_slopes_nan_guard() -> None:
    values = np.stack([np.full((2, 1), 1.0), np.full((2, 1), np.nan)],
                      axis=-1)
    slopes, n_bad = ols_slopes(values)
    assert n_bad == 2
    assert (slopes == 0.0).all()


def test_trend_slopes_min_points_rule() -> None:
    cal = MonthCalendar.make(1979, 200)
    sic = np.zeros((4, 3, 200))
    # March needs >= 10 occurrences strictly before the issue
    issue_10th = cal.index_of(1988, 4)  # 10 Marches before (1979..1988)
    slopes = trend_slopes(sic, cal.months, issue_10th, 3)
    assert (slopes == 0.0).all()
    issue_9th = cal.index_of(1987, 4)   # only 9 Marches before
    slopes = trend_slopes(sic, cal.months, issue_9th, 3)
    assert (slopes == 0.0).all()


def test_trend_slopes_causal_and_trailing30() -> None:
    cal = MonthCalendar.make(1979, 600)
    months = cal.months
    # constant ramp in May only: +1 per year from May 1979 on
    sic = np.zeros((2, 2, cal.n_months))
    for i, m in enumerate(months):
        if m == 5:
            sic[:, :, i] = float(i // 12)
    issue = cal.index_of(2010, 6)  # June 2010: 31 Mays before
    slopes = trend_slopes(sic, months, issue, 5)
    assert np.allclose(slopes, 1.0)
    # the trailing-30 window: perturbing the 31st-oldest May must not matter
    idx_may = np.where(months[:issue] == 5)[0]
    sic2 = sic.copy()
    sic2[:, :, idx_may[0]] = 100.0
    slopes2 = trend_slopes(sic2, months, issue, 5)
    assert np.allclose(slopes2, 1.0)
    # the target calendar month differs from other months: April untouched
    slopes_apr = trend_slopes(sic, months, issue, 4)
    assert (slopes_apr == 0.0).all()
