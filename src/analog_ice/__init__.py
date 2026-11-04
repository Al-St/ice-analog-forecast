"""analog-ice: deterministic seasonal sea-ice analog forecasting.

A small, self-contained implementation of monthly-case analog forecasting
of Arctic sea-ice concentration with trend-corrected transfer:

- fixed calendar anchors (March maximum / September minimum) define the
  history window of every issue month;
- a same-calendar-month candidate pool (at least 7 months back) provides
  the analogs; retrieval is a freshness-weighted euclidean distance over
  window values and first differences, top-k uniform;
- each analog field is transferred along the calendar-identical offset
  and corrected per cell with a causal trailing-30 OLS slope of the
  target calendar month before averaging;
- an optional multi-season scale reuses the same machinery on
  same-month interannual trajectories (leads 12/24/36).

Deterministic: no random numbers, float64 throughout until the float32
release cast, stable tie-breaks in ranking.
"""
from .calendar import MonthCalendar
from .features import FeatureFrames, Frame, fit_pca
from .forecast import ForecastResult, run_scale1, run_scale2
from .inputs import (
    load_latitude, load_ocean_mask, load_sic_work_array,
    monthly_sic_from_cdr,
)
from .metrics import anomaly_correlation, iiee, mae_mse, summarize
from .regions import RegionMasks, load_region_masks, pan_arctic_extent, \
    regional_extents
from .retrieval import CaseFeatures, case_features, pair_distance, \
    rank_analogs, same_month_pool
from .serialize import load_run, payload, save_run, sha256_file
from .trend import ols_slopes, trend_slopes
from .transfer import Scale2Trajectory, scale2_distance, transfer_scale1, \
    transfer_scale2
from .windows import ANCHOR_LAGS, Window, anchor_lag, slot_weights, \
    window_for_month

__version__ = "1.0.0"

__all__ = [
    "MonthCalendar", "FeatureFrames", "Frame", "fit_pca",
    "ForecastResult", "run_scale1", "run_scale2",
    "load_latitude", "load_ocean_mask", "load_sic_work_array",
    "monthly_sic_from_cdr",
    "anomaly_correlation", "iiee", "mae_mse", "summarize",
    "RegionMasks", "load_region_masks", "pan_arctic_extent",
    "regional_extents",
    "CaseFeatures", "case_features", "pair_distance", "rank_analogs",
    "same_month_pool",
    "load_run", "payload", "save_run", "sha256_file",
    "ols_slopes", "trend_slopes",
    "Scale2Trajectory", "scale2_distance", "transfer_scale1",
    "transfer_scale2",
    "ANCHOR_LAGS", "Window", "anchor_lag", "slot_weights", "window_for_month",
    "__version__",
]
