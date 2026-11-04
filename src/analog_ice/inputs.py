"""Input loading and the NSIDC CDR conversion recipes.

Two byte-exact conversion recipes are implemented here (they are part of
the reproducibility contract and are unit-tested):

1. Monthly SIC work array from the NSIDC CDR monthly file
   (``cdr_seaice_conc_monthly``, uint8 percent with flag values > 100):
   read the raw bytes with netCDF4 auto mask/scale DISABLED, take the
   months of the scientific record, set every flagged value (raw > 100)
   to 0.0, rescale ``float32(raw) * float32(0.01)``, and store as
   float64 with the month axis last: shape (448, 304, N).

2. Ocean mask: a cell is land iff the CDR land flag (raw == 254) is set.
   The land-flag set is not perfectly static: three months of the record
   flag parts of the land as missing instead, so the mask is read from
   the LAST month of the record (its 254 set is the superset covering
   every other month). Ocean = not land (72,485 ocean cells on the
   448 x 304 grid).

The regional predictor series (18 NSIDC sea-ice regions) and the
pan-Arctic extent series built from them live in :mod:`analog_ice.regions`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

LAND_FLAG = 254  # CDR uint8 flag: land (static across months)
FLAG_CUTOFF = 100  # raw values above this are flags, not concentrations
CDR_VARIABLE = "cdr_seaice_conc_monthly"


def load_sic_work_array(path: Path) -> np.ndarray:
    """Load the monthly SIC work array, float64 (H, W, N), month axis last.

    The canonical array is produced by :func:`monthly_sic_from_cdr` (or
    was prepared once from it); this function only validates the layout.
    """
    sic = np.load(Path(path))
    if sic.ndim != 3 or sic.dtype != np.float64:
        raise ValueError(f"expected float64 (H, W, N), got {sic.dtype} {sic.shape}")
    return sic


def load_ocean_mask(path: Path) -> np.ndarray:
    """Load the ocean mask (bool, (H, W)); True = ocean."""
    ocean = np.load(Path(path))
    if ocean.dtype != np.bool_ or ocean.ndim != 2:
        raise ValueError(f"expected bool (H, W), got {ocean.dtype} {ocean.shape}")
    return ocean


def load_latitude(path: Path) -> np.ndarray:
    """Load the 2-D latitude grid (float64, (H, W))."""
    lat = np.load(Path(path))
    if lat.ndim != 2:
        raise ValueError(f"expected (H, W) latitude, got {lat.shape}")
    return lat.astype(np.float64)


def monthly_sic_from_cdr(nc_path: Path,
                         start_year: int,
                         n_months: int) -> tuple[np.ndarray, np.ndarray]:
    """Monthly SIC (H, W, N) float64 + ocean mask (H, W) bool from the CDR
    monthly file, using the byte-exact conversion recipes of this module.

    The record must start at January of ``start_year``; the CDR file begins
    two months earlier (November of ``start_year - 1``), so the scientific
    record starts at CDR index 2.
    """
    with Dataset(str(nc_path)) as ds:
        ds.set_auto_maskandscale(False)  # flags must survive as raw uint8
        raw = np.array(ds.variables[CDR_VARIABLE][:], dtype=np.uint8)

    # This CDR product starts in November of start_year - 1, i.e. exactly
    # two months before the scientific record.
    first_idx = 2
    if raw.shape[0] < first_idx + n_months:
        raise ValueError(
            f"CDR record too short: {raw.shape[0]} months "
            f"(< {first_idx + n_months})")
    window = raw[first_idx:first_idx + n_months]

    conc = window.astype(np.float32) * np.float32(0.01)
    conc[window > FLAG_CUTOFF] = np.float32(0.0)
    sic = np.moveaxis(conc.astype(np.float64), 0, -1)

    # Land flag of the LAST month: its 254 set is the superset of every
    # other month's (three months flag parts of the land as missing).
    land = window[-1] == LAND_FLAG
    ocean = ~land
    return sic, ocean


def latitude_from_cdr(nc_path: Path) -> np.ndarray:
    """2-D latitude grid from the CDR file, float64 (H, W)."""
    with Dataset(str(nc_path)) as ds:
        lat = np.array(ds.variables["latitude"][:], dtype=np.float64)
    return lat
