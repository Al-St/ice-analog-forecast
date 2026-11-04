"""Case features, the same-month candidate pool, and analog retrieval.

Because the candidate pool is restricted to months with the SAME calendar
month as the issue (at least 7 months earlier, so that all six lead
solutions are observable), query and candidate windows always have the
same depth and phase; the distance is a freshness-weighted euclidean over
values and first differences (the phase term is identically zero within a
pool and is kept only for formula uniformity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Frame
from .windows import Window, slot_weights, window_for_month

N_ANALOGS = 5


@dataclass
class CaseFeatures:
    """Feature view of one case (issue month + its anchored window)."""

    index: int                # record position of the issue month
    window: Window
    values: np.ndarray        # (n_slots, n_series) feature values
    diffs: np.ndarray         # (n_slots - 1, n_series) first differences
    phase: tuple[float, float]  # (cos, sin) of 2*pi*cal_month/12


def case_features(index: int, frame: Frame, months: np.ndarray) -> CaseFeatures:
    """Window, values, first differences and the phase pair of a case.

    Values/differences are read from the frame's rows at the window slot
    positions (anchor-first ordering).
    """
    window = window_for_month(index, months)
    values = frame.scores[list(window.slots)]
    diffs = np.diff(values, axis=0)
    cal = int(months[index])
    angle = 2.0 * np.pi * cal / 12.0
    phase = (float(np.cos(angle)), float(np.sin(angle)))
    return CaseFeatures(index=index, window=window, values=values,
                        diffs=diffs, phase=phase)


def same_month_pool(issue_idx: int, months: np.ndarray,
                    base_years: int = 0) -> list[int]:
    """Candidate pool of an issue: every record position ``t_c`` with the
    same calendar month as the issue and ``t_c <= issue_idx - 7`` (so all
    six lead transfers stay inside the record and the analog years are
    distinct from the evaluation period).

    ``base_years`` = L > 0 restricts the pool to the trailing 12*L months
    (``t_c > issue_idx - 7 - 12*L``).
    """
    lo = 0 if not base_years else max(0, issue_idx - 7 - 12 * base_years + 1)
    cal = int(months[issue_idx])
    return [t for t in range(lo, issue_idx - 7 + 1)
            if int(months[t]) == cal]


def pair_distance(query: CaseFeatures, cand: CaseFeatures) -> float:
    """Squared euclidean distance between two cases of the same phase.

    All window slots are common by construction; if a record-truncated
    candidate has fewer slots, the common TRAILING slots are compared with
    the query's weights at their true slot positions (deeper query slots
    drop out, they are not penalized). The phase pair of equal months
    contributes exactly zero.
    """
    n = min(len(query.values), len(cand.values))
    w = slot_weights(query.window.lag, n)
    dv = query.values[-n:] - cand.values[-n:]
    d2 = float((w[:, None] * dv * dv).sum())
    if n > 1:
        dd = query.diffs[-(n - 1):] - cand.diffs[-(n - 1):]
        d2 += float((w[1:][:, None] * dd * dd).sum())
    d2 += (query.phase[0] - cand.phase[0]) ** 2 \
        + (query.phase[1] - cand.phase[1]) ** 2
    return d2


def rank_analogs(query: CaseFeatures, candidates: list[CaseFeatures],
                 k: int = N_ANALOGS) -> tuple[list[CaseFeatures], list[float]]:
    """Top-k candidates by ascending distance.

    Deterministic tie-break: (distance, record position). Uniform analog
    weights (the mean over the top-k is taken downstream).
    """
    distances = [pair_distance(query, c) for c in candidates]
    order = sorted(range(len(candidates)),
                   key=lambda i: (distances[i], candidates[i].index))
    top = order[:k]
    return [candidates[i] for i in top], [distances[i] for i in top]
