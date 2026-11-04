"""End-to-end smoke test on a small synthetic record: both arms run,
shapes are right, and repeated runs are byte-identical (determinism)."""
from __future__ import annotations

import numpy as np

from analog_ice import (FeatureFrames, MonthCalendar, load_run, run_scale1,
                        run_scale2, save_run)

N_MONTHS = 420  # 1979-01 .. 2013-12


def _setup():
    calendar = MonthCalendar.make(1979, N_MONTHS)
    months = calendar.months
    rng = np.random.default_rng(123)
    # two correlated predictor series with a seasonal cycle
    seasonal = np.cos(2.0 * np.pi * (months - 3) / 12.0)
    s1 = seasonal * 3.0 + 0.1 * rng.standard_normal(N_MONTHS)
    s2 = 0.5 * s1 + 0.2 * rng.standard_normal(N_MONTHS)
    series = np.column_stack([s1, s2])
    # concentration fields driven by the predictors (so analogs matter)
    h, w = 4, 3
    sic = np.zeros((h, w, N_MONTHS))
    base = 0.4 + 0.1 * (series @ np.array([[0.3], [0.2]]))[:, 0]
    for i in range(N_MONTHS):
        sic[:, :, i] = np.clip(base[i] + 0.02 * rng.standard_normal((h, w)),
                               0.0, 1.0)
    return calendar, sic, series


def test_scale1_runs_and_is_deterministic() -> None:
    calendar, sic, series = _setup()
    frames = FeatureFrames(series, calendar.months, mode="raw",
                           pca="identity")
    targets = list(range(calendar.index_of(2005, 1),
                         calendar.index_of(2005, 4) + 1))
    result = run_scale1(sic, calendar, frames, targets)
    assert result.preds.shape == (4, 4, 3, 6)
    assert result.truths.shape == result.preds.shape
    assert result.preds.dtype == np.float32
    assert np.isfinite(result.preds).all()
    assert (result.preds >= 0.0).all() and (result.preds <= 1.0).all()
    assert result.target_labels[0] == "2005-01"
    # truth equals the input field (repeated over the lead columns)
    for pos, t in enumerate(targets):
        assert np.array_equal(result.truths[pos, :, :, 0],
                              sic[:, :, t].astype(np.float32))
        assert np.array_equal(result.truths[pos, :, :, 3],
                              sic[:, :, t].astype(np.float32))
    # determinism: rebuild everything and compare bytes
    frames2 = FeatureFrames(series, calendar.months, mode="raw",
                            pca="identity")
    result2 = run_scale1(sic, calendar, frames2, targets)
    assert np.array_equal(result.preds, result2.preds)
    # one diagnostic per issue month
    assert len(result.diagnostics) == len({t - l for t in targets
                                           for l in range(1, 7)})


def test_scale1_base_years_limits_pool() -> None:
    calendar, sic, series = _setup()
    frames = FeatureFrames(series, calendar.months, mode="raw",
                           pca="identity")
    targets = [calendar.index_of(2010, 9)]
    full = run_scale1(sic, calendar, frames, targets, base_years=0)
    limited = run_scale1(sic, calendar, frames, targets, base_years=10)
    n_full = [d["n_candidates"] for d in full.diagnostics]
    n_limited = [d["n_candidates"] for d in limited.diagnostics]
    assert n_limited[0] == 10
    assert n_full[0] > n_limited[0]


def test_scale2_runs_and_is_deterministic() -> None:
    calendar, sic, series = _setup()
    frames = FeatureFrames(series, calendar.months, mode="raw",
                           pca="identity")
    issues = list(range(calendar.index_of(2009, 1),
                        calendar.index_of(2009, 3) + 1))
    result = run_scale2(sic, calendar, frames, issues)
    assert result.preds.shape == (3, 4, 3, 3)
    assert result.lead_months == (12, 24, 36)
    assert np.isfinite(result.preds).all()
    assert np.array_equal(result.target_labels[0],
                          np.array(["2010-01", "2011-01", "2012-01"],
                                   dtype="<U7"))
    frames2 = FeatureFrames(series, calendar.months, mode="raw",
                            pca="identity")
    result2 = run_scale2(sic, calendar, frames2, issues)
    assert np.array_equal(result.preds, result2.preds)


def test_save_and_load_round_trip(tmp_path) -> None:
    calendar, sic, series = _setup()
    frames = FeatureFrames(series, calendar.months, mode="raw",
                           pca="identity")
    targets = [calendar.index_of(2008, 3), calendar.index_of(2008, 4)]
    result = run_scale1(sic, calendar, frames, targets)
    npz_path, digest = save_run(result, tmp_path / "out",
                                metadata={"arm": "a"})
    loaded = load_run(npz_path)
    assert np.array_equal(loaded["preds"], result.preds)
    assert loaded["target_ym"][0] == "2008-03"
    # a second save of the same payload produces the identical file bytes
    _, digest2 = save_run(result, tmp_path / "out2", metadata={"arm": "a"})
    assert digest == digest2
