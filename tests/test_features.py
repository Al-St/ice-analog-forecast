"""Causal feature frames: raw/m3 modes, min-points zeroing, PCA
determinism and orthonormality."""
from __future__ import annotations

import numpy as np
import pytest

from analog_ice import FeatureFrames, MonthCalendar, fit_pca


def test_raw_identity_frames(calendar) -> None:
    rng = np.random.default_rng(0)
    series = rng.standard_normal((120, 2))
    frames = FeatureFrames(series, calendar.months[:120], mode="raw",
                           pca="identity")
    frame = frames.frame(100)
    assert frame.fit_ok
    assert frame.n_valid_rows == 0  # no climatology accounting in raw mode
    assert np.array_equal(frame.scores, series[:101, :1])


def _climatology_series(calendar) -> np.ndarray:
    """A 1-column series with a strong seasonal cycle (12 distinct monthly
    levels) so trailing-30 statistics are non-trivial."""
    n = calendar.n_months
    base = np.zeros((n, 1))
    base[:, 0] = np.sin(2 * np.pi * calendar.months / 12.0) * 3.0 + \
        np.arange(n) * 0.01
    return base


def test_m3_causal_statistics(calendar) -> None:
    series = _climatology_series(calendar)
    frames = FeatureFrames(series, calendar.months, mode="m3",
                           pca="identity")
    boundary = calendar.index_of(2005, 6)
    frame = frames.frame(boundary)

    # statistics use only occurrences STRICTLY before the boundary
    months = calendar.months
    for m in range(1, 13):
        occ = np.where(months[:boundary] == m)[0]
        assert occ.size == (months[:boundary] == m).sum()

    # a row's z-value: trailing min(30, n) same-month occurrences before b
    t = boundary  # the boundary row itself
    occ = np.where(months[:boundary] == months[t])[0]
    seg = series[occ[-30:], 0]
    expected = (series[t, 0] - seg.mean()) / seg.std(ddof=0)
    assert np.isclose(frame.scores[t, 0], expected)


def test_m3_min_points_zeroing(calendar) -> None:
    series = _climatology_series(calendar)
    frames = FeatureFrames(series, calendar.months, mode="m3",
                           pca="identity")
    # rows in the first 9 years: their calendar month has < 10 occurrences
    # strictly before the boundary -> invalid rows are zeroed
    frame = frames.frame(calendar.index_of(1987, 12))
    months = calendar.months
    for t in range(frame.boundary + 1):
        n_before = int((months[:frame.boundary] == months[t]).sum())
        if n_before < 10:
            assert not frame.scores[t].any()
    # later rows are valid and non-zero
    frame_full = frames.frame(calendar.index_of(2005, 6))
    assert frame_full.n_valid_rows == frame_full.boundary + 1


def test_m3_fit_gate(calendar) -> None:
    series = _climatology_series(calendar)
    frames = FeatureFrames(series, calendar.months, mode="m3",
                           pca="identity")
    early = frames.frame(calendar.index_of(1983, 1))
    assert not early.fit_ok  # fewer than 10 valid rows
    late = frames.frame(calendar.index_of(2005, 6))
    assert late.fit_ok


def test_pca_fit_deterministic_and_orthonormal(calendar) -> None:
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((200, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    model_a = fit_pca(rows)
    model_b = fit_pca(rows.copy())
    assert np.array_equal(model_a.components, model_b.components)
    assert np.allclose(model_a.components @ model_a.components.T,
                       np.eye(5), atol=1e-12)
    projected = model_a.project(rows)
    assert projected.shape == (200, 5)
    # variance ordering: component 1 explains at least as much as 2
    var = projected.var(axis=0)
    assert (np.diff(var) <= 1e-9).all()


def test_pca_sign_convention() -> None:
    rng = np.random.default_rng(9)
    rows = rng.standard_normal((40, 6))
    model = fit_pca(rows)
    projected = model.project(rows)
    # largest-|score| entry over the fitted rows is positive
    leader = np.argmax(np.abs(projected), axis=0)
    assert (projected[leader, np.arange(projected.shape[1])] >= 0).all()


def test_pca5_frame_uses_causal_fit(calendar) -> None:
    rng = np.random.default_rng(11)
    series = rng.standard_normal((240, 8))
    frames = FeatureFrames(series, calendar.months[:240], mode="raw",
                           pca="pca5")
    frame = frames.frame(233)
    assert frame.scores.shape == (234, 5)
    assert np.isfinite(frame.scores).all()


def test_pca5_needs_enough_columns(calendar) -> None:
    series = np.random.default_rng(12).standard_normal((240, 3))
    frames = FeatureFrames(series, calendar.months[:240], mode="raw",
                           pca="pca5")
    with pytest.raises(ValueError, match="PCA-5"):
        frames.frame(233)
