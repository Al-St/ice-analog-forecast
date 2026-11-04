"""Transfer with trend correction (scale 1) and the scale-2 trajectory
view."""
from __future__ import annotations

import numpy as np
import pytest

from analog_ice import MonthCalendar, Scale2Trajectory, scale2_distance, \
    transfer_scale1, transfer_scale2
from analog_ice.features import FeatureFrames


def test_transfer_scale1_plain_mean() -> None:
    """Two analogs transfer 5 months forward; the prediction is the plain
    mean of the transferred fields (analogs are <= issue - 7, so the
    transferred months stay strictly before the issue)."""
    sic = np.zeros((2, 2, 40))
    sic[:, :, 10] = 0.2   # analog 5 transfers here (5 + lead 5)
    sic[:, :, 15] = 0.4   # analog 10 transfers here (10 + lead 5)
    cal = MonthCalendar.make(1979, 40)
    beta = np.zeros((2, 2))
    field = transfer_scale1([5, 10], 22, 17, sic, cal.years, beta)
    assert np.allclose(field, 0.3)


def test_transfer_scale1_trend_correction_and_clip() -> None:
    """Each analog field is shifted by beta * dy (years between the
    transferred month and the target) and clipped to [0, 1]."""
    cal = MonthCalendar.make(1979, 40)
    sic = np.zeros((1, 1, 40))
    sic[0, 0, 24] = 0.5   # analog July 1980 transfers to January 1981
    years = cal.years
    # issue July 1981 (idx 30), target January 1982 (idx 36), lead 6
    beta = np.full((1, 1), 0.1)
    field = transfer_scale1([18], 36, 30, sic, years, beta)
    expected = np.clip(0.5 + 0.1 * (1982 - 1981), 0.0, 1.0)
    assert np.allclose(field, expected)
    # clipping at 1.0
    beta_hi = np.full((1, 1), 0.5)
    field_hi = transfer_scale1([18], 36, 30, sic, years, beta_hi)
    assert np.allclose(field_hi, 1.0)


def test_transfer_scale1_out_of_record_guard() -> None:
    """The transfer refuses a source month at/after the issue (the same-
    month pool normally makes this unreachable)."""
    cal = MonthCalendar.make(1979, 41)
    sic = np.zeros((1, 1, 41))
    with pytest.raises(AssertionError):
        transfer_scale1([28], 40, 34, sic, cal.years,
                        np.zeros((1, 1)))  # src = 34 == issue


def _frames_from_scores(calendar, scores: np.ndarray) -> FeatureFrames:
    frames = FeatureFrames(scores, calendar.months)
    return frames


def test_scale2_trajectory_ordering(calendar) -> None:
    scores = np.arange(calendar.n_months, dtype=np.float64)[:, None]
    frames = FeatureFrames(scores, calendar.months)
    boundary = calendar.index_of(2010, 4)
    frame = frames.frame(boundary)
    traj = Scale2Trajectory.build(frame.scores, calendar.months, boundary, 4)
    # positions (y, April), freshest first: the 31 Aprils before the
    # boundary provide 30 trajectory values (the oldest is the differencing
    # anchor only)
    idx_all = np.where(calendar.months[:boundary] == 4)[0]
    take = idx_all[-31:][::-1]
    vals = frame.scores[take]
    assert traj.n_years == 30
    assert np.array_equal(traj.values[:, 0], vals[:-1, 0])
    assert np.allclose(traj.diffs[:, 0], vals[:-1, 0] - vals[1:, 0])
    # age weights: freshest = 1, oldest = 1/30
    w = traj.weights()
    assert w[0] == 1.0 and np.isclose(w[-1], 1 / 30)


def test_scale2_trajectory_min_years(calendar) -> None:
    scores = np.zeros((calendar.n_months, 1))
    frames = FeatureFrames(scores, calendar.months)
    early = calendar.index_of(1984, 4)
    with pytest.raises(ValueError):
        Scale2Trajectory.build(frames.frame(early).scores, calendar.months,
                               early, 4)


def test_scale2_distance_zero_for_identical() -> None:
    rng = np.random.default_rng(5)
    values = rng.standard_normal((30, 2))
    traj = Scale2Trajectory(values, values - np.roll(values, 1, axis=0), 30)
    assert scale2_distance(traj, traj) == pytest.approx(0.0)


def test_scale2_distance_common_ages(calendar) -> None:
    """A deep-age difference does not change the distance when the
    candidate has fewer years (deep ages dropped, not penalized)."""
    rng = np.random.default_rng(6)
    q_vals = rng.standard_normal((30, 2))
    q = Scale2Trajectory(q_vals, rng.standard_normal((30, 2)), 30)
    c_vals = q_vals.copy()
    c = Scale2Trajectory(c_vals, np.zeros((30, 2)), 30)
    base = scale2_distance(q, c)
    c_short = Scale2Trajectory(c_vals[:10], np.zeros((10, 2)), 10)
    short = scale2_distance(q, c_short)
    assert base > 0.0 and short > 0.0 and short < base


def test_transfer_scale2_plus_one_year(calendar) -> None:
    sic = np.zeros((1, 1, calendar.n_months))
    years = calendar.years
    sic[0, 0, calendar.index_of(2000, 3)] = 0.7
    beta = np.zeros((1, 1))
    field = transfer_scale2([1999], 2001, 3, calendar.index_of, sic, beta)
    # candidate year 1999 transfers March 2000 (+1 year), dy = 2001-2000
    assert np.allclose(field, 0.7)
