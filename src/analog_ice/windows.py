"""Fixed calendar anchors and history windows (the v2.1 window rule).

Every calendar month is anchored at a FIXED climatological anchor month:
the maximum-extent month March (``A_max``) or the minimum-extent month
September (``A_min``). The lag ``d`` from the anchor is a constant of the
calendar month (0..5), so the history window ``[anchor, m0]`` is
calendar-deterministic: no recursion, no clipping, no unanchored cases.

The only irregularity is the record start: January/February candidates of
the first record year have their anchor before the record, so their
window is truncated to the available months (fewer slots); slot alignment
stays anchor-based (the truncated candidate contributes only its trailing
slots — see :func:`analog_ice.retrieval.pair_distance`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANCHOR_MAX_MONTH = 3  # March: climatological maximum month
ANCHOR_MIN_MONTH = 9  # September: climatological minimum month

# calendar month -> (anchor type, lag d in months back from the issue)
ANCHOR_LAGS: dict[int, tuple[str, int]] = {
    3: ("max", 0), 4: ("max", 1), 5: ("max", 2),
    6: ("max", 3), 7: ("max", 4), 8: ("max", 5),
    9: ("min", 0), 10: ("min", 1), 11: ("min", 2),
    12: ("min", 3), 1: ("min", 4), 2: ("min", 5),
}


@dataclass(frozen=True)
class Window:
    """A history window of one case.

    Attributes:
        slots: record positions of the window, anchor-first .. issue.
        lag: anchor lag d (len(slots) is d + 1 for a full window).
        anchor_type: ``"max-03"`` / ``"min-09"`` label of the anchor.
    """

    slots: tuple[int, ...]
    lag: int
    anchor_type: str


def anchor_lag(cal_month: int) -> tuple[str, int]:
    """(anchor type, lag d) of a calendar month: January/February anchor at
    September of the previous year; d counts months back from the issue."""
    return ANCHOR_LAGS[int(cal_month)]


def window_for_month(idx: int, months: np.ndarray) -> Window:
    """Window ``[anchor, issue]`` of record position ``idx`` (d months back
    to the fixed anchor; truncated at the record start)."""
    cal = int(months[idx])
    kind, d = anchor_lag(cal)
    anchor_idx = idx - d
    lo = max(0, anchor_idx)
    label = (f"max-{ANCHOR_MAX_MONTH:02d}" if kind == "max"
             else f"min-{ANCHOR_MIN_MONTH:02d}")
    return Window(slots=tuple(range(lo, idx + 1)), lag=d, anchor_type=label)


def slot_weights(lag: int, n_slots: int) -> np.ndarray:
    """Linear freshness weights of the LAST ``n_slots`` slots of a window
    of true depth ``lag``: ``w(s) = (s + 1) / (lag + 1)`` evaluated at the
    true slot positions s = months from the anchor (s = lag is the issue).
    A full window (n_slots = lag + 1) gets the plain increasing profile;
    a truncated candidate window keeps the weights of its own trailing
    positions (deeper query slots drop out, they are not penalized)."""
    if not 0 <= n_slots <= lag + 1:
        raise ValueError(f"n_slots {n_slots} incompatible with lag {lag}")
    s0 = lag - n_slots + 1
    return np.array([(s + 1) / (lag + 1) for s in range(s0, lag + 1)],
                    dtype=np.float64)
