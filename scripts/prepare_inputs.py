#!/usr/bin/env python
"""Prepare the model inputs from the NSIDC CDR monthly file.

Outputs (into --out-dir):
  sic_monthly.npy         float64 (448, 304, N), months --start-year-01 ..
  ocean_mask.npy          bool (448, 304), True = ocean (CDR land flag 254)
  latitude.npy            float64 (448, 304)

Byte-exact recipes (unit-tested, see tests/test_inputs.py):
- concentrations: raw uint8 percent read with netCDF auto mask/scale off;
  flagged values (raw > 100) -> 0.0; float32(raw) * float32(0.01);
- ocean: not(land flag 254); the flag field is static across the record.

Optionally verifies against existing work arrays with --check paths.

Usage:
  python scripts/prepare_inputs.py --cdr <cdr_monthly.nc> \
      --start-year 1979 --n-months 540 --out-dir <work dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analog_ice.inputs import (  # noqa: E402
    latitude_from_cdr, monthly_sic_from_cdr,
)


def xarray_coordinate_check(nc_path: Path, n_time: int, n_y: int,
                            n_x: int) -> None:
    """Cross-check the file's coordinate variables with xarray (declared
    dependency; the flag-accurate raw read itself must use netCDF4 with
    auto mask/scale disabled)."""
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        time_size = ds.sizes.get("tdim")
        y_size = ds.sizes.get("y")
        x_size = ds.sizes.get("x")
    if (time_size, y_size, x_size) != (n_time, n_y, n_x):
        raise ValueError(
            f"coordinate sizes {(time_size, y_size, x_size)} disagree "
            f"with the data array {(n_time, n_y, n_x)}")
    print(f"[prepare] xarray coordinate check: tdim={time_size} "
          f"y={y_size} x={x_size} OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdr", required=True, type=Path,
                    help="NSIDC CDR monthly concentration .nc (uint8 flags)")
    ap.add_argument("--start-year", type=int, default=1979)
    ap.add_argument("--n-months", type=int, default=540,
                    help="months of the scientific record (1979-01..2023-12 "
                         "= 540)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--check-sic", type=Path, default=None,
                    help="optional existing monthly array to compare bitwise")
    ap.add_argument("--check-ocean", type=Path, default=None,
                    help="optional existing ocean mask to compare bitwise")
    args = ap.parse_args()

    print(f"[prepare] reading {args.cdr}", flush=True)
    sic, ocean = monthly_sic_from_cdr(args.cdr, args.start_year,
                                      args.n_months)
    xarray_coordinate_check(args.cdr, sic.shape[2] + 2, *sic.shape[:2])
    lat = latitude_from_cdr(args.cdr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sic_path = args.out_dir / "sic_monthly.npy"
    ocean_path = args.out_dir / "ocean_mask.npy"
    lat_path = args.out_dir / "latitude.npy"
    np.save(sic_path, sic)
    np.save(ocean_path, ocean)
    np.save(lat_path, lat)
    print(f"[prepare] sic {sic.shape} {sic.dtype} -> {sic_path}")
    print(f"[prepare] ocean mask: {int(ocean.sum())} ocean cells "
          f"({100.0 * ocean.mean():.2f}%) -> {ocean_path}")
    print(f"[prepare] latitude {lat.shape} -> {lat_path}")

    if args.check_sic:
        ref = np.load(args.check_sic)
        same = np.array_equal(ref, sic)
        print(f"[prepare] SIC bitwise equal to {args.check_sic}: {same}")
        if not same:
            diff = np.abs(ref - sic)
            print(f"[prepare]   max|d| = {float(diff.max()):.6g}")
    if args.check_ocean:
        ref = np.load(args.check_ocean)
        same = np.array_equal(ref.astype(bool), ocean)
        print(f"[prepare] ocean mask bitwise equal to {args.check_ocean}: "
              f"{same}")
    return


if __name__ == "__main__":
    main()
