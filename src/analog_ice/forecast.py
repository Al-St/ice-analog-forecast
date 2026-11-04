"""Forecast orchestration: scale 1 (monthly leads 1..6) and scale 2
(multi-season leads 12/24/36).

Both scales share the same deterministic pipeline: causal feature frames
(one per distinct issue/candidate boundary), the same-month pool, the
freshness-weighted retrieval of k = 5 analogs (ties broken by record
position), and the trend-corrected transfer of the analog fields.

No random numbers anywhere; float64 throughout until the float32 cast of
the assembled prediction fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .calendar import MonthCalendar
from .features import FeatureFrames
from .retrieval import N_ANALOGS, case_features, rank_analogs, same_month_pool
from .transfer import (
    SCALE2_K, SCALE2_LEADS_DEFAULT, SCALE2_MIN_YEARS,
    Scale2Trajectory, scale2_distance, transfer_scale1, transfer_scale2,
)
from .trend import WINDOW, trend_slopes

SCALE1_LEADS = (1, 2, 3, 4, 5, 6)


@dataclass
class ForecastResult:
    """Assembled predictions of one run plus retrieval diagnostics.

    Attributes:
        preds: float32 fields (n_rows, H, W, n_leads), no NaNs.
        truths: float32 verifying fields at the targets.
        target_labels: ``YYYY-MM`` of every target row (scale 2: per lead).
        issue_labels: ``YYYY-MM`` of the issuing month of every row.
        lead_months: lead lengths of the last axis.
        diagnostics: one record per issue month (pool size, analog months,
            distances).
    """

    preds: np.ndarray
    truths: np.ndarray
    target_labels: np.ndarray
    issue_labels: np.ndarray
    lead_months: tuple[int, ...]
    diagnostics: list[dict] = field(default_factory=list)


def run_scale1(sic: np.ndarray,
               calendar: MonthCalendar,
               frames: FeatureFrames,
               target_indices: Sequence[int],
               leads: Sequence[int] = SCALE1_LEADS,
               base_years: int = 0,
               k: int = N_ANALOGS,
               trend_window: int = WINDOW) -> ForecastResult:
    """Scale-1 forecast: every (target, lead) pair is produced by the issue
    month ``target - lead``.

    Args:
        sic: (H, W, N) monthly concentration array, float64.
        calendar: the record calendar.
        frames: causal feature frames (one series configuration per arm).
        target_indices: record positions to predict.
        leads: monthly leads (default 1..6).
        base_years: same-month pool length limit in years (0 = full base).
        k: number of analogs (uniform weights).
    """
    months = calendar.months
    years = calendar.years
    target_set = set(target_indices)
    issues = sorted({t - l for t in target_indices for l in leads})
    fields_by_issue: dict[int, dict[int, np.ndarray]] = {}
    diagnostics: list[dict] = []
    beta_cache: dict[tuple[int, int], np.ndarray] = {}

    for m in issues:
        query_frame = frames.frame(m)
        if not query_frame.fit_ok:
            raise ValueError(f"query frame at {m} has no valid PCA fit")
        query = case_features(m, query_frame, months)
        pool = same_month_pool(m, months, base_years)
        candidates = []
        n_excluded = 0
        for t in pool:
            frame = frames.frame(t)
            if not frame.fit_ok:
                n_excluded += 1
                continue
            candidates.append(case_features(t, frame, months))
        if len(candidates) < k:
            raise ValueError(f"pool {len(candidates)} < k={k} at issue {m}")

        analogs, distances = rank_analogs(query, candidates, k)
        analog_indices = [c.index for c in analogs]
        cal = int(months[m])
        if not all(int(months[t]) == cal for t in analog_indices):
            raise AssertionError(f"cross-phase analogs at issue {m}")

        t_list = [m + l for l in leads if (m + l) in target_set]
        fields: dict[int, np.ndarray] = {}
        for t in t_list:
            key = (m, int(months[t]))
            if key not in beta_cache:
                beta_cache[key] = trend_slopes(sic, months, m, key[1],
                                               window=trend_window)
            fields[t] = transfer_scale1(analog_indices, t, m, sic, years,
                                        beta_cache[key])
        fields_by_issue[m] = fields

        diagnostics.append({
            "issue_label": calendar.label(m),
            "cal_month": cal,
            "lag": query.window.lag,
            "anchor_type": query.window.anchor_type,
            "n_pool": len(pool),
            "n_candidates": len(candidates),
            "n_excluded_nofit": n_excluded,
            "analogs": [calendar.label(t) for t in analog_indices],
            "distances": [float(d) for d in distances],
        })

    n_leads = len(leads)
    preds = np.full((len(target_indices), *sic.shape[:2], n_leads), np.nan,
                    dtype=np.float32)
    truths = np.full_like(preds, np.nan)
    for t_pos, t in enumerate(target_indices):
        truth = sic[:, :, t].astype(np.float32)
        for l_pos in range(n_leads):
            truths[t_pos, :, :, l_pos] = truth
        for l_pos, lead in enumerate(leads):
            m = t - lead
            fields = fields_by_issue.get(m)
            if fields is not None and t in fields:
                preds[t_pos, :, :, l_pos] = fields[t].astype(np.float32)
    if np.isnan(preds).any():
        raise AssertionError("NaN left in predictions")

    target_labels = np.array([calendar.label(t) for t in target_indices],
                             dtype="<U7")
    issue_labels = np.array(
        [[calendar.label(t - l) for l in leads] for t in target_indices],
        dtype="<U7")
    return ForecastResult(preds=preds, truths=truths,
                          target_labels=target_labels,
                          issue_labels=issue_labels,
                          lead_months=tuple(int(l) for l in leads),
                          diagnostics=diagnostics)


def run_scale2(sic: np.ndarray,
               calendar: MonthCalendar,
               frames: FeatureFrames,
               issue_indices: Sequence[int],
               leads: Sequence[int] = SCALE2_LEADS_DEFAULT,
               k: int = N_ANALOGS) -> ForecastResult:
    """Scale-2 forecast: same-month interannual trajectories; leads are
    multi-season (12/24/36 months). The candidate set of an issue are the
    calendar YEARS ``y' <= Y_i - 2``, each with its own frame; the
    transferred month is ``(y' + 1, m_T)`` (the "+1 year" shift).
    """
    months = calendar.months
    preds = np.full((len(issue_indices), *sic.shape[:2], len(leads)),
                    np.nan, dtype=np.float32)
    truths = np.full_like(preds, np.nan)
    diagnostics: list[dict] = []

    for r, m in enumerate(issue_indices):
        cal_month = int(months[m])  # = target month-of-year
        query_traj = Scale2Trajectory.build(frames.frame(m).scores, months,
                                            m, cal_month)
        year_i = int(calendar.years[m])

        cand_entries = []
        n_dropped = 0
        for y_p in range(calendar.start_year, year_i - 1):  # y' <= Y_i - 2
            b = calendar.index_of(y_p, cal_month)
            frame = frames.frame(b)
            if not frame.fit_ok:
                n_dropped += 1
                continue
            try:
                traj = Scale2Trajectory.build(frame.scores, months, b,
                                              cal_month)
            except ValueError:
                n_dropped += 1
                continue
            cand_entries.append((y_p, traj))
        distances = [scale2_distance(query_traj, traj)
                     for _, traj in cand_entries]
        order = sorted(range(len(cand_entries)),
                       key=lambda i: (distances[i], cand_entries[i][0]))
        top = order[:k]
        analog_years = [cand_entries[i][0] for i in top]

        beta = trend_slopes(sic, months, m, cal_month)
        for l_pos, lead in enumerate(leads):
            t = m + lead
            if t >= calendar.n_months:
                raise AssertionError((m, lead))
            target_year = int(calendar.years[t])
            field = transfer_scale2(analog_years, target_year, cal_month,
                                    calendar.index_of, sic, beta)
            preds[r, :, :, l_pos] = field.astype(np.float32)
            truths[r, :, :, l_pos] = sic[:, :, t].astype(np.float32)

        diagnostics.append({
            "issue_label": calendar.label(m),
            "n_candidates": len(cand_entries),
            "n_dropped": n_dropped,
            "k_query": query_traj.n_years,
            "analogs": analog_years,
            "distances": [float(distances[i]) for i in top],
        })

    target_labels = np.array(
        [[calendar.label(m + l) for l in leads] for m in issue_indices],
        dtype="<U7")
    issue_labels = np.array([calendar.label(m) for m in issue_indices],
                            dtype="<U7")
    return ForecastResult(preds=preds, truths=truths,
                          target_labels=target_labels,
                          issue_labels=issue_labels,
                          lead_months=tuple(int(l) for l in leads),
                          diagnostics=diagnostics)
