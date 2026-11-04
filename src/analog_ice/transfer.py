"""Transfer of analog fields to the target month, with the trend
correction applied BEFORE averaging.

Scale 1 (leads tau = 1..6): the analog year offsets carry over
identically — a same-month candidate ``t_c`` maps to ``t_c + tau``, whose
calendar month equals the target's by construction. Each analog field is
corrected per cell with the trailing-30 OLS slope of the target calendar
month (``dy`` = years between the transferred month and the target),
clipped to [0, 1], and the prediction is the plain mean over the analogs.

Scale 2 (leads tau in {12, 24, 36}): candidates are calendar YEARS (the
same-month interannual trajectory view); the "+1 year" shift lives here
only — candidate year ``y'`` maps to month ``(y' + 1, m_T)``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .trend import trend_slopes

SCALE2_K = 30
SCALE2_MIN_YEARS = 10
SCALE2_LEADS_DEFAULT = (12, 24, 36)


def transfer_scale1(analog_indices: Sequence[int],
                    target_idx: int,
                    issue_idx: int,
                    sic: np.ndarray,
                    years: np.ndarray,
                    beta: np.ndarray) -> np.ndarray:
    """Mean over analogs of ``clip01(X(t_c + tau) + beta * dy)``.

    ``beta`` are the cell-wise slopes for the TARGET calendar month at
    this issue (see :func:`analog_ice.trend.trend_slopes`); ``dy`` is the
    year distance between the transferred month ``t_c + tau`` and the
    target.
    """
    lead = target_idx - issue_idx
    acc = np.zeros(sic.shape[:2], dtype=np.float64)
    for t_c in analog_indices:
        src = t_c + lead
        if not 0 <= src < issue_idx:
            raise AssertionError((t_c, src, issue_idx))
        dy = int(years[target_idx]) - int(years[src])
        acc += np.clip(sic[:, :, src] + beta * dy, 0.0, 1.0)
    return acc / len(analog_indices)


class Scale2Trajectory:
    """Same-month interannual trajectory of one case (scale 2 view).

    Values are the case frame's rows at (y, m_T) for the K years ending
    with the last occurrence of ``m_T`` strictly before the boundary,
    freshest first, plus their year-to-year differences. Raises
    ``ValueError`` when fewer than ``SCALE2_MIN_YEARS`` usable years are
    available (the caller drops the case).
    """

    __slots__ = ("values", "diffs", "n_years")

    def __init__(self, values: np.ndarray, diffs: np.ndarray,
                 n_years: int) -> None:
        self.values = values
        self.diffs = diffs
        self.n_years = n_years

    @staticmethod
    def build(frame_scores: np.ndarray, months: np.ndarray,
              boundary: int, cal_month: int, k: int = SCALE2_K,
              min_years: int = SCALE2_MIN_YEARS) -> "Scale2Trajectory":
        """Trajectory of the case with frame rows ``t <= boundary``.

        Values at positions (y, cal_month), y descending from the last
        occurrence of ``cal_month`` strictly before ``boundary``; K' =
        min(k, available - 1) years; differences are newer-minus-older.
        """
        idx_all = np.where(months[:boundary] == int(cal_month))[0]
        n_rows = min(k + 1, len(idx_all))
        if n_rows - 1 < min_years:
            raise ValueError(
                f"boundary {boundary}: only {n_rows - 1} usable years "
                f"(< {min_years})")
        take = idx_all[-n_rows:][::-1]  # freshest first
        vals = frame_scores[take]
        values = vals[:-1]
        diffs = vals[:-1] - vals[1:]
        return Scale2Trajectory(values, diffs, int(len(values)))

    def weights(self) -> np.ndarray:
        """Linear age weights w(a) = (K' - a) / K', a = 0 freshest."""
        return np.array([(self.n_years - a) / self.n_years
                         for a in range(self.n_years)], dtype=np.float64)


def scale2_distance(query: Scale2Trajectory,
                    cand: Scale2Trajectory) -> float:
    """Weighted euclidean over common ages; ONE weight per age multiplies
    both the value and the difference term; deeper candidate ages are
    dropped, not penalized. The phase pair (month-of-year) is constant
    within one target and contributes zero."""
    qv, qd, qw = query.values, query.diffs, query.weights()
    cv, cd = cand.values, cand.diffs
    n = min(len(qv), len(cv))
    w = qw[:n]
    dv = qv[:n] - cv[:n]
    d2 = float((w[:, None] * dv * dv).sum())
    if n > 1:
        dd = qd[:n - 1] - cd[:n - 1]
        d2 += float((w[:n - 1][:, None] * dd * dd).sum())
    return d2


def transfer_scale2(analog_years: Sequence[int],
                    target_year: int,
                    cal_month: int,
                    calendar_index_of,
                    sic: np.ndarray,
                    beta: np.ndarray) -> np.ndarray:
    """Mean over analog YEARS of ``clip01(X(y' + 1, m_T) + beta * dy)``.

    ``calendar_index_of(year, month)`` resolves the record position of the
    transferred month ``(y' + 1, m_T)``; ``dy`` = target year minus the
    transferred year. (This is the scale-2 "+1 year" shift.)
    """
    acc = np.zeros(sic.shape[:2], dtype=np.float64)
    for y_p in analog_years:
        src = calendar_index_of(y_p + 1, cal_month)
        dy = target_year - (y_p + 1)
        acc += np.clip(sic[:, :, src] + beta * dy, 0.0, 1.0)
    return acc / len(analog_years)
