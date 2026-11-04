"""Same-month pool, distances with freshness weights, deterministic
ranking, truncated candidates."""
from __future__ import annotations

import numpy as np
import pytest

from analog_ice import (FeatureFrames, MonthCalendar, case_features,
                        pair_distance, rank_analogs, same_month_pool)


def test_same_month_pool(calendar) -> None:
    months = calendar.months
    idx = calendar.index_of(2016, 3)   # March issue
    pool = same_month_pool(idx, months, base_years=0)
    assert pool == [t for t in range(0, idx - 6)
                    if months[t] == 3]
    assert all(months[t] == 3 for t in pool)
    assert max(pool) == idx - 12       # same month => <= idx - 12
    assert len(pool) == 2016 - 1979    # March 1979..2015

    # base_years=10 restricts to the trailing 120 months
    pool10 = same_month_pool(idx, months, base_years=10)
    assert pool10 == [t for t in pool if t > idx - 7 - 120]
    assert len(pool10) == 10           # March 2006..2015


def test_pair_distance_zero_for_identical(calendar) -> None:
    rng = np.random.default_rng(3)
    series = rng.standard_normal((300, 1))
    frames = FeatureFrames(series, calendar.months[:300])
    idx = calendar.index_of(1998, 4)
    a = case_features(idx, frames.frame(idx), calendar.months)
    assert pair_distance(a, a) == 0.0


def test_pair_distance_freshness_weights(calendar) -> None:
    """With identical series, a perturbation at the freshest slot (weight
    1) outweighs the same perturbation at the deepest slot (weight
    1/(d+1))."""
    months = calendar.months
    constant = np.ones((calendar.n_months, 1))
    frames_a = FeatureFrames(constant, months)
    frames_b = FeatureFrames(constant.copy(), months)
    frames_c = FeatureFrames(constant.copy(), months)

    issue = calendar.index_of(2010, 6)             # lag 3 (max anchor)
    other = calendar.index_of(2005, 6)             # same phase, lag 3
    query = case_features(issue, frames_a.frame(issue), months)

    # perturb the candidate's deepest (anchor) slot only
    anchor = frames_b.series
    anchor[other - 3] += 10.0
    cand_deep = case_features(other, frames_b.frame(other), months)
    # perturb the candidate's freshest (issue) slot only
    frames_c.series[other] += 10.0
    cand_fresh = case_features(other, frames_c.frame(other), months)

    d_deep = pair_distance(query, cand_deep)
    d_fresh = pair_distance(query, cand_fresh)
    # deep: value term 100 * w0 = 25 plus diff term 100 * w1 = 50
    assert d_deep == pytest.approx(75.0)
    # fresh: value term 100 * w3 = 100 plus diff term 100 * w3 = 100
    assert d_fresh == pytest.approx(200.0)
    assert d_fresh > d_deep > 0.0


def test_rank_analogs_tie_break(calendar) -> None:
    months = calendar.months
    constant = np.ones((calendar.n_months, 1)) * 2.0
    frames = FeatureFrames(constant, months)
    issue = calendar.index_of(2010, 5)
    query = case_features(issue, frames.frame(issue), months)
    pool = same_month_pool(issue, months, 0)
    candidates = [case_features(t, frames.frame(t), months) for t in pool]
    analogs, distances = rank_analogs(query, candidates, k=5)
    # all distances tie at 0 -> the deterministic (distance, position)
    # tie-break keeps the five OLDEST candidates
    assert distances == [0.0] * 5
    idxs = [c.index for c in analogs]
    assert idxs == pool[:5]


def test_truncated_candidate_trailing_slots() -> None:
    """A record-start candidate with fewer slots contributes only its
    trailing slots, weighted at the query's true positions."""
    cal = MonthCalendar.make(1979, 480)
    months = cal.months
    series = np.arange(cal.n_months, dtype=np.float64)[:, None]
    frames = FeatureFrames(series, months)
    issue = cal.index_of(1998, 1)              # January: lag 4
    query = case_features(issue, frames.frame(issue), months)
    cand = case_features(0, frames.frame(0), months)  # Jan 1979: 1 slot
    n = min(len(query.values), len(cand.values))
    assert n == 1
    # only the freshest slots are compared; weight = (lag+1)/(lag+1) = 1
    expected = (query.values[-1] - cand.values[-1]) ** 2 * 1.0
    assert pair_distance(query, cand) == pytest.approx(expected)
