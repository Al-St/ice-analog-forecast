"""Metrics on small hand-checked arrays (MAE / RMSE / IIEE / ACC)."""
from __future__ import annotations

import numpy as np

from analog_ice import anomaly_correlation, iiee, mae_mse, summarize

OCEAN = np.array([[True, False], [True, True]])


def test_mae_mse_hand_checked() -> None:
    # one row, two leads; two ocean cells only: (0,0) and (1,1)
    ocean = np.array([[True, False], [False, True]])
    preds = np.zeros((1, 2, 2, 2))
    preds[0, 0, 0, 0] = 0.3   # obs 0.1 -> |0.2|
    preds[0, 1, 1, 0] = 0.0   # obs 0.5 -> |0.5|
    preds[0, 0, 0, 1] = 0.2   # obs 0.1 -> |0.1|
    preds[0, 1, 1, 1] = 0.6   # obs 0.5 -> |0.1|
    truths = np.zeros((1, 2, 2, 2))
    truths[0, 0, 0, :] = 0.1
    truths[0, 1, 1, :] = 0.5
    mae, mse = mae_mse(preds, truths, ocean)
    assert np.isclose(mae[0, 0], (0.2 + 0.5) / 2)
    assert np.isclose(mae[1, 1], (0.1 + 0.1) / 2)   # lead 1, month 1
    assert np.isclose(mse[0, 0], (0.04 + 0.25) / 2)
    # land cells ignored
    preds[0, 0, 1, 0] = 100.0
    mae2, _ = mae_mse(preds, truths, ocean)
    assert np.isclose(mae2[0, 0], (0.2 + 0.5) / 2)


def test_iiee_symmetric_difference_area() -> None:
    # threshold 0.15; obs edge on (0,0) only; pred edge on (0,0) and (1,1)
    preds = np.zeros((1, 2, 2, 1))
    preds[0, 0, 0, 0] = 0.5
    preds[0, 1, 1, 0] = 0.5
    truths = np.zeros_like(preds)
    truths[0, 0, 0, 0] = 0.5
    area, oe, ue = iiee(preds, truths, OCEAN)
    # XOR cells on the ocean: (1,1) only -> 1 * 625 km2 = 6.25e-4 1e6 km2
    assert np.isclose(area[0, 0], 625.0 / 1e6)
    assert np.isclose(oe[0, 0], 625.0 / 1e6)   # predicted-only
    assert np.isclose(ue[0, 0], 0.0)


def test_iiee_perfect_zero() -> None:
    field = np.zeros((2, 2, 2, 1))
    field[:, 0, 0, 0] = 0.5
    area, oe, ue = iiee(field, field.copy(), OCEAN)
    assert (area == 0).all() and (oe == 0).all() and (ue == 0).all()


def test_acc_perfect_and_anticorrelated() -> None:
    rng = np.random.default_rng(1)
    obs = rng.random((10, 2, 2, 1))
    acc = anomaly_correlation(obs, obs.copy(), OCEAN)
    assert np.isclose(acc[0], 1.0)
    # exact anticorrelation: anomalies of p = -o_anom
    o_anom = obs - obs.mean(axis=0, keepdims=True)
    preds = 0.5 - o_anom
    truths = 0.5 + o_anom
    acc_anti = anomaly_correlation(preds, truths, OCEAN)
    assert np.isclose(acc_anti[0], -1.0)


def test_summarize_keys_and_finiteness(synthetic_sic) -> None:
    ocean = np.ones(synthetic_sic.shape[:2], dtype=bool)
    preds = synthetic_sic[:, :, :12].transpose(2, 0, 1)[..., None]
    preds = np.repeat(preds, 2, axis=-1)
    truths = synthetic_sic[:, :, :12].transpose(2, 0, 1)[..., None]
    truths = np.repeat(truths, 2, axis=-1)
    summary = summarize(preds, truths, ocean)
    assert set(summary) == {"mae_mean", "rmse_mean", "iiee_mean",
                            "acc_mean"}
    assert summary["mae_mean"] == 0.0
    assert summary["iiee_mean"] == 0.0
    assert np.isclose(summary["acc_mean"], 1.0)
