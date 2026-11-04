#!/usr/bin/env python
"""Verify a produced run against a stored reference run.

Compares, in order of strictness:
  1. metric_summary.json values (rounded to 4 and 6 decimals);
  2. predictions.npz arrays bitwise (preds and truths);
  3. the SHA-256 of the npz files.

Exits 0 when the configured success criterion holds, 1 otherwise
(printing a verdict report).

Usage:
  python scripts/verify_reference.py --produced outputs/arm_a_full \
      --reference <ref_dir> [--ocean ocean_mask.npy] [--require-bitwise]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analog_ice import load_run, sha256_file  # noqa: E402
from analog_ice.metrics import summarize  # noqa: E402

METRIC_KEYS = ("mae_mean", "rmse_mean", "iiee_mean", "acc_mean")


def find_npz(root: Path) -> Path:
    """predictions.npz either directly in the dir or one level below
    (the reference layout nests it under target_month_export/<model>/)."""
    direct = root / "predictions.npz"
    if direct.is_file():
        return direct
    for nested in sorted(root.glob("**/predictions.npz")):
        return nested
    raise FileNotFoundError(f"no predictions.npz under {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--produced", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--ocean", required=True, type=Path)
    ap.add_argument("--require-bitwise", action="store_true",
                    help="fail unless the npz arrays AND the file digest "
                         "are identical")
    args = ap.parse_args()

    produced_npz = find_npz(args.produced)
    reference_npz = find_npz(args.reference)
    print(f"[verify] produced  : {produced_npz}")
    print(f"[verify] reference : {reference_npz}")

    mine = load_run(produced_npz)
    ref = load_run(reference_npz)

    report: dict = {"produced": str(produced_npz),
                    "reference": str(reference_npz)}

    arrays_equal = {k: bool(np.array_equal(mine[k], ref[k]))
                    for k in mine}
    report["arrays_bitwise_equal"] = arrays_equal
    diffs = {}
    for k in ("preds", "truths"):
        if not arrays_equal[k]:
            d = np.abs(mine[k].astype(np.float64) - ref[k].astype(np.float64))
            diffs[k] = {"max_abs_diff": float(d.max()),
                        "n_different_cells": int((d > 0).sum())}
    report["diff_stats"] = diffs
    print(f"[verify] arrays bitwise equal: {arrays_equal}")
    if diffs:
        print(f"[verify] diff stats: {json.dumps(diffs)}")

    sha_mine = sha256_file(produced_npz)
    sha_ref = sha256_file(reference_npz)
    report["sha256_equal"] = bool(sha_mine == sha_ref)
    report["sha256_produced"] = sha_mine
    report["sha256_reference"] = sha_ref
    print(f"[verify] sha256 equal: {report['sha256_equal']}")

    ocean = np.load(args.ocean).astype(bool)
    metrics_mine = summarize(mine["preds"], mine["truths"], ocean)
    ref_summary_path = None
    for candidate in (args.reference / "metric_summary.json",
                      *args.reference.glob("**/metric_summary.json")):
        if candidate.is_file():
            ref_summary_path = candidate
            break
    metrics_match = {}
    if ref_summary_path is not None:
        with open(ref_summary_path, encoding="utf-8") as handle:
            ref_metrics = json.load(handle)
        for key in METRIC_KEYS:
            if key not in ref_metrics:
                continue
            ref_val = float(ref_metrics[key])
            mine_val = float(metrics_mine[key])
            metrics_match[key] = {
                "reference": ref_val,
                "produced": mine_val,
                "equal_6dp": round(mine_val, 6) == round(ref_val, 6),
                "equal_4dp": round(mine_val, 4) == round(ref_val, 4),
                "abs_diff": abs(mine_val - ref_val),
            }
    report["metrics"] = metrics_match
    print(json.dumps(metrics_match, indent=2))

    metrics_ok_4dp = bool(metrics_match) and \
        all(v["equal_4dp"] for v in metrics_match.values())
    bitwise_ok = all(arrays_equal.values())
    sha_ok = report["sha256_equal"]
    if args.require_bitwise:
        verdict = bitwise_ok and sha_ok
        criterion = "bitwise arrays + identical file digest"
    else:
        verdict = metrics_ok_4dp
        criterion = "metrics equal to 4 decimals"
    report["criterion"] = criterion
    report["verdict"] = "PASS" if verdict else "FAIL"

    out_path = args.produced / "verify_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[verify] criterion ({criterion}): "
          f"{'PASS' if verdict else 'FAIL'}")
    print(f"[verify] report -> {out_path}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
