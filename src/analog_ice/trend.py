"""Causal cell-wise linear trend correction of the transferred fields.

For every (issue, target calendar month) pair, an OLS slope per grid cell
is estimated over the trailing ``window`` most recent occurrences of that
calendar month STRICTLY BEFORE the issue month (consecutive occurrences,
so the slope is per year). Cells with any NaN in the window get slope 0.
When fewer than ``min_points`` occurrences exist the correction is
disabled (slopes = 0).
"""
from __future__ import annotations

import numpy as np

MIN_POINTS = 10
WINDOW = 30


def ols_slopes(values: np.ndarray) -> tuple[np.ndarray, int]:
    """OLS slope per cell of ``values`` (H, W, n): each cell's series along
    the last axis regressed on the occurrence index 0..n-1 (consecutive
    occurrences -> the slope is per year).

    NaN guard: a cell with ANY NaN in the window gets slope 0 (and is
    counted). Returns (slopes (H, W) float64, number of guarded cells).
    """
    n = values.shape[-1]
    xc = np.arange(n, dtype=np.float64)
    xc -= xc.mean()
    sxx = float((xc * xc).sum())
    bad = np.isnan(values).any(axis=-1)
    if bad.any():
        values = np.where(bad[..., None], 0.0, values)
    slopes = np.einsum("i,hwi->hw", xc, values) / sxx
    slopes[bad] = 0.0
    return slopes, int(bad.sum())


def trend_slopes(sic: np.ndarray, months: np.ndarray, issue_idx: int,
                 cal_month: int, window: int = WINDOW,
                 min_points: int = MIN_POINTS) -> np.ndarray:
    """Cell-wise OLS slopes for ONE (issue, target calendar month) pair.

    Uses only occurrences of ``cal_month`` strictly before ``issue_idx``:
    the trailing ``min(window, n)`` of them. Fewer than ``min_points``
    occurrences disable the correction (all-zero slopes).
    """
    idx = np.where(np.asarray(months[:issue_idx]) == int(cal_month))[0]
    if idx.size < min_points:
        return np.zeros(sic.shape[:2], dtype=np.float64)
    idx = idx[-int(window):]
    slopes, _ = ols_slopes(sic[:, :, idx])
    return slopes
