"""Verification metrics on the (rows, H, W, leads) prediction stacks.

Conventions (fixed; the numbers in the paper use exactly these):

- MAE / RMSE: per (lead, target calendar month) mean absolute / squared
  error over finite ocean cells; the reported scalar means over the
  (leads, 12) matrices. RMSE = sqrt(mean of the monthly MSE matrix).
- IIEE (sea-ice edge error): per (lead, month) area of the symmetric
  difference of the binary SIC > 0.15 maps, 625 km2 per cell, reported in
  10^6 km2; scalar = mean over the matrix.
- ACC: per lead, per ocean cell, Pearson correlation of the interannual
  anomalies over the row axis (the mean over rows is removed per cell);
  scalar = nanmean over ocean cells and leads.

Inputs are cast to float32 first (the released arrays are float32).
"""
from __future__ import annotations

import numpy as np

SIC_THRESHOLD = 0.15
CELL_AREA_KM2 = 625.0


def mae_mse(preds: np.ndarray, truths: np.ndarray, ocean: np.ndarray,
            start_month_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(leads, 12) MAE and MSE matrices over ocean cells.

    ``start_month_index`` is the calendar-month index (0 = January) of the
    first row; rows are consecutive months.
    """
    preds = preds.astype(np.float32)
    truths = truths.astype(np.float32)
    n_leads = preds.shape[-1]
    mae = np.zeros((n_leads, 12), dtype=np.float64)
    mse = np.zeros((n_leads, 12), dtype=np.float64)
    counts = np.zeros((n_leads, 12), dtype=np.float64)
    for i in range(preds.shape[0]):
        for j in range(n_leads):
            month = (start_month_index + i + j) % 12
            obs = truths[i, :, :, j][ocean]
            sim = preds[i, :, :, j][ocean]
            valid = np.isfinite(obs) & np.isfinite(sim)
            if not np.any(valid):
                continue
            mae[j, month] += np.mean(np.abs(obs[valid] - sim[valid]))
            mse[j, month] += np.mean((obs[valid] - sim[valid]) ** 2)
            counts[j, month] += 1.0
    counts = np.where(counts > 0, counts, 1.0)
    return mae / counts, mse / counts


def iiee(preds: np.ndarray, truths: np.ndarray,
         start_month_index: int = 0,
         threshold: float = SIC_THRESHOLD,
         cell_area_km2: float = CELL_AREA_KM2,
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(leads, 12) IIEE / overestimate / underestimate areas in 10^6 km2.

    Binary edge maps are ``field > threshold``; IIEE is the symmetric
    difference area, OE the predicted-only area, UE the observed-only.
    """
    preds = preds.astype(np.float32)
    truths = truths.astype(np.float32)
    n_leads = preds.shape[-1]
    iiee_sum = np.zeros((n_leads, 12), dtype=np.float64)
    oe_sum = np.zeros_like(iiee_sum)
    ue_sum = np.zeros_like(iiee_sum)
    counts = np.zeros_like(iiee_sum)
    for i in range(preds.shape[0]):
        for j in range(n_leads):
            month = (start_month_index + i + j) % 12
            obs = (truths[i, :, :, j] > threshold).astype(np.int32)
            sim = (preds[i, :, :, j] > threshold).astype(np.int32)
            union = sim + obs
            union[union == 2] = 1
            inter = sim * obs
            diff_area = union - inter          # == 1 exactly on XOR cells
            oe_area = sim - inter
            ue_area = obs - inter
            iiee_sum[j, month] += np.sum(diff_area == 1) * cell_area_km2 / 1e6
            oe_sum[j, month] += np.sum(oe_area == 1) * cell_area_km2 / 1e6
            ue_sum[j, month] += np.sum(ue_area == 1) * cell_area_km2 / 1e6
            counts[j, month] += 1.0
    counts = np.where(counts > 0, counts, 1.0)
    return iiee_sum / counts, oe_sum / counts, ue_sum / counts


def anomaly_correlation(preds: np.ndarray, truths: np.ndarray,
                        ocean: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-lead ACC: per ocean cell, Pearson correlation of the anomalies
    over the row axis; nanmean over ocean cells per lead."""
    preds = preds.astype(np.float32)
    truths = truths.astype(np.float32)
    n_leads = preds.shape[-1]
    acc = np.full(n_leads, np.nan, dtype=np.float64)
    for c in range(n_leads):
        p = preds[:, :, :, c]
        o = truths[:, :, :, c]
        p_anom = p - np.mean(p, axis=0, keepdims=True)
        o_anom = o - np.mean(o, axis=0, keepdims=True)
        num = np.sum(p_anom * o_anom, axis=0)
        den = np.sqrt(np.sum(p_anom ** 2, axis=0)
                      * np.sum(o_anom ** 2, axis=0)) + eps
        r_grid = np.where(den > eps, num / den, np.nan)
        acc[c] = np.nanmean(r_grid[ocean])
    return acc


def summarize(preds: np.ndarray, truths: np.ndarray, ocean: np.ndarray,
              start_month_index: int = 0) -> dict[str, float]:
    """The four headline scalars: MAE / RMSE / IIEE / ACC."""
    mae, mse = mae_mse(preds, truths, ocean, start_month_index)
    iiee_area, _, _ = iiee(preds, truths, start_month_index)
    acc = anomaly_correlation(preds, truths, ocean)
    return {
        "mae_mean": float(np.mean(mae)),
        "rmse_mean": float(np.sqrt(np.mean(mse))),
        "iiee_mean": float(np.mean(iiee_area)),
        "acc_mean": float(np.nanmean(acc)),
    }
