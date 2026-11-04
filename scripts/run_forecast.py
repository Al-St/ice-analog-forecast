#!/usr/bin/env python
"""Run the analog forecast: inputs -> predictions.npz + metric summary.

Arms:
  a   scalar arm: pan-Arctic extent series (sum of the 18 regional
      extents), raw values, no decomposition;
  b   regional arm: 18 regional extent series, causal anomalization
      (m3) and PCA-5 (arm a is the minimal arm; arm b needs the NSIDC
      region raster, see data_format.md).

Lead scales:
  --scale 1           monthly leads 1..6 (default), targets 2016-01..
                      2023-12;
  --scale seasonal    same-month interannual trajectories, leads 12/24/36,
                      issues 2016-01..2020-12 (regional-arm features).
                      This is the paper's "interannual scale" (main.tex
                      sec:method:scale2, campaign arm h); the CLI keeps
                      the historical name "seasonal" so stored run
                      directories and SHAs stay valid, and accepts
                      "interannual" as an exact alias (canonicalized to
                      "seasonal", so metadata and outputs are identical).

Pool length: --base-years L restricts the candidate pool to the trailing
12*L months (0 = full record base).

Inputs default to files inside --work (see data_format.md):
  sic_monthly.npy | ocean_mask.npy | latitude.npy | regions_nsidc0780.nc

Usage:
  python scripts/run_forecast.py --work <data dir> --arm a \
      --base-years 0 --out outputs/arm_a_full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analog_ice import (  # noqa: E402
    MonthCalendar, FeatureFrames, load_ocean_mask, load_sic_work_array,
    load_region_masks, pan_arctic_extent, regional_extents,
    run_scale1, run_scale2, save_run, summarize,
)

TARGET_START = (2016, 1)
TARGET_END = (2023, 12)          # inclusive, scale 1
SCALE2_ISSUE_START = (2016, 1)
SCALE2_ISSUE_END = (2020, 12)    # inclusive
START_YEAR = 1979
N_MONTHS = 540

ARMS = {
    "a": {"mode": "raw", "pca": "identity", "needs_regions": True},
    "b": {"mode": "m3", "pca": "pca5", "needs_regions": True},
}

# CLI spellings for the 12/24/36-month scale.  "seasonal" is the
# canonical value (historical: stored run dirs, metadata.json, SHAs);
# "interannual" is the manuscript's term (main.tex sec:method:scale2)
# and is accepted as an exact alias, canonicalized to "seasonal" so a
# run produces byte-identical metadata and predictions either way.
SCALE_CHOICES = ("1", "seasonal", "interannual")
SCALE_ALIASES = {"interannual": "seasonal"}


def canonical_scale(value: str) -> str:
    """Map a --scale CLI value to its canonical spelling."""
    if value not in SCALE_CHOICES:
        raise ValueError(f"unknown scale {value!r}; expected one of "
                         f"{SCALE_CHOICES}")
    return SCALE_ALIASES.get(value, value)


def month_range(calendar: MonthCalendar, start: tuple[int, int],
                end: tuple[int, int]) -> list[int]:
    a = calendar.index_of(*start)
    b = calendar.index_of(*end)
    return list(range(a, b + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path,
                    help="directory with sic_monthly.npy, ocean_mask.npy, "
                         "latitude.npy, regions_nsidc0780.nc")
    ap.add_argument("--arm", choices=sorted(ARMS), default="a")
    ap.add_argument("--base-years", type=int, default=0,
                    help="candidate pool length limit in years (0 = full)")
    ap.add_argument("--scale", choices=SCALE_CHOICES, default="1",
                    help="1 = monthly leads 1..6; seasonal (= interannual, "
                         "the paper's term) = leads 12/24/36")
    ap.add_argument("--analogs", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sic", type=Path, default=None)
    ap.add_argument("--ocean", type=Path, default=None)
    ap.add_argument("--regions", type=Path, default=None)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()
    args.scale = canonical_scale(args.scale)

    cfg = ARMS[args.arm]
    sic_path = args.sic or (args.work / "sic_monthly.npy")
    ocean_path = args.ocean or (args.work / "ocean_mask.npy")
    regions_path = args.regions or (args.work / "regions_nsidc0780.nc")

    t0 = time.time()
    print(f"[run] arm={args.arm} scale={args.scale} "
          f"base_years={args.base_years or 'full'}", flush=True)
    sic = load_sic_work_array(sic_path)
    ocean = load_ocean_mask(ocean_path)
    if sic.shape[:2] != ocean.shape:
        raise ValueError("ocean mask and SIC grids disagree")
    calendar = MonthCalendar.make(START_YEAR, sic.shape[2])

    print("[run] loading regions ...", flush=True)
    regions = load_region_masks(regions_path, ocean)
    print(f"[run] {len(regions.ids)} non-empty regions", flush=True)

    if args.arm == "a":
        series = pan_arctic_extent(sic, regions)[:, None]
    else:
        series = regional_extents(sic, regions)
    frames = FeatureFrames(series, calendar.months,
                           mode=cfg["mode"], pca=cfg["pca"])

    if args.scale == "1":
        targets = month_range(calendar, TARGET_START, TARGET_END)
        result = run_scale1(sic, calendar, frames, targets,
                            base_years=args.base_years, k=args.analogs)
    else:
        issues = month_range(calendar, SCALE2_ISSUE_START, SCALE2_ISSUE_END)
        result = run_scale2(sic, calendar, frames, issues, k=args.analogs)

    metadata = {
        "model": "analog-ice",
        "version": "1.0.0",
        "arm": args.arm,
        "scale": args.scale,
        "base_years": args.base_years or None,
        "features": {
            "series": ("pan_arctic_extent (sum of the 18 regional extents)"
                       if args.arm == "a" else
                       "18 regional extents (monthly_sie convention)"),
            "mode": cfg["mode"],
            "pca": cfg["pca"],
        },
        "retrieval": {"k": args.analogs, "weights": "uniform",
                      "pool": f"same-calendar-month, t_c <= t_i - 7"
                              + (f", trailing {12 * args.base_years} months"
                                 if args.base_years else "")},
        "trend": {"mode": "cellwise trailing OLS", "window": 30,
                  "min_points": 10, "clip": [0.0, 1.0]},
        "n_rows": int(result.preds.shape[0]),
        "n_issues": len(result.diagnostics),
    }
    out_dir = args.out
    npz_path, digest = save_run(result, out_dir, metadata)
    print(f"[run] wrote {npz_path}", flush=True)
    print(f"[run] predictions.npz SHA256 = {digest}", flush=True)
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump({"n_issues": len(result.diagnostics),
                   "issues": result.diagnostics}, handle, indent=2)
    if not args.no_metrics:
        metrics = summarize(result.preds, result.truths, ocean)
        metrics = {k: round(v, 6) for k, v in metrics.items()}
        metrics["npz_sha256"] = digest
        with open(out_dir / "metric_summary.json", "w",
                  encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2), flush=True)
    print(f"[run] done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
