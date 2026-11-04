# ice-analog-forecast

Deterministic seasonal forecasting of Arctic sea-ice concentration by
monthly-case analog retrieval with a causal trend-corrected transfer.
Self-contained: only numpy/scipy/netCDF4/xarray, no framework code, no
random numbers, one entry point per run.

## The method in five lines

1. Every issue month has a history window anchored at a FIXED calendar
   anchor — the climatological March maximum (months March..August) or
   September minimum (months September..February), lag d = 0..5.
2. Predictors are monthly sea-ice extents: 18 regional extents (NSIDC
   sea-ice regions), or their pan-Arctic sum; optionally causally
   anomalized (trailing-30 same-month statistics, N >= 10) and reduced
   by PCA-5 fitted causally at every case.
3. Candidates come from a same-calendar-month pool at least 7 months
   back; retrieval ranks them by a freshness-weighted euclidean distance
   over window values and first differences (weights `(s+1)/(d+1)`),
   top-k = 5, uniform.
4. Each analog field transfers along the calendar-identical offset and
   is corrected per cell by a causal trailing-30 OLS slope of the target
   calendar month (`dy` years, clip [0,1]) BEFORE the analog mean.
5. A multi-season scale reuses the same machinery on same-month
   interannual trajectories (leads 12/24/36, the "+1 year" shift).

## Terminology

The manuscript and this repository name the two lead scales
differently; the mapping is one-to-one:

| this repository | manuscript (`main.tex`) | campaign arm |
|---|---|---|
| scale 1 (`--scale 1`), monthly leads 1..6 | the monthly scale | arms a, b and their ablations |
| "seasonal" / "multi-season" scale (`--scale seasonal`), leads 12/24/36 | the **interannual scale** (§ "The interannual scale: 12--36-month horizons", `\label{sec:method:scale2}`) | arm h |

The CLI value stays `seasonal` so stored run directories,
`metadata.json` records and SHA-256 digests remain valid;
`--scale interannual` is accepted as an exact alias and canonicalized
to `seasonal`, producing byte-identical metadata and predictions either
way. In this repository "seasonal" never refers to a within-year
seasonal cycle: candidates always come from a same-calendar-month pool,
and the name only distinguishes the 12/24/36-month interannual scheme
from the monthly leads 1..6.

## Installation

Python 3.10 and:

```
pip install -r requirements.txt   # or: pip install -e .
```

numpy/scipy versions are pinned: the PCA decomposition uses
`scipy.linalg.svd` with a deterministic sign convention; changing BLAS
versions can change SVD output and break bitwise reproducibility.

## Quick start

Prepare the inputs once (see `data_format.md` for the array formats):

```
python scripts/prepare_inputs.py --cdr <cdr_monthly.nc> \
    --start-year 1979 --n-months 540 --out-dir <data dir>
cp <nsidc0780_regions.nc> <data dir>/regions_nsidc0780.nc
```

Run the scalar arm (pan-Arctic extent, raw values) on the full record
base, leads 1..6, targets 2016-01..2023-12:

```
python scripts/run_forecast.py --work <data dir> --arm a --base-years 0 \
    --out outputs/arm_a_full
```

Regional arm (18 extents + causal anomalization + PCA-5), and the
10-year sliding base dose point:

```
python scripts/run_forecast.py --work <data dir> --arm b --base-years 0 \
    --out outputs/arm_b_full
python scripts/run_forecast.py --work <data dir> --arm b --base-years 10 \
    --out outputs/arm_b_L10
```

Interannual scale (= `seasonal` in the CLI; leads 12/24/36, issues
2016-01..2020-12, regional features; see Terminology above):

```
python scripts/run_forecast.py --work <data dir> --arm b --scale seasonal \
    --out outputs/seasonal
```

Each run writes `predictions.npz` (preds, truths, target_ym, issue_ym,
lead_months), `metadata.json`, `diagnostics.json` and — unless
`--no-metrics` — `metric_summary.json` with MAE / RMSE / IIEE (10^6
km^2, 0.15 edge threshold) / ACC (anomaly correlation over ocean cells).

## Data

`data_format.md` documents the expected arrays, their provenance and the
byte-exact NSIDC CDR conversion recipes (concentration rescaling, land
flag 254 -> ocean mask). The region raster is the NSIDC-0780 Sea Ice
Regions file (polar stereographic 25 km, same 448 x 304 grid). All three
scale-1 arms need it: the regional arm uses the 18 extents, the scalar
arm uses their sum (the regions cover the peripheral seas and the
central Arctic; unassigned ocean, raster flag 0, is excluded — so the
sum over regions is NOT the total-ocean extent).

## Reproducibility

No RNG anywhere; float64 throughout until the float32 release cast;
distance ties break by (distance, record position); the npz payload and
its key order are canonical. `scripts/verify_reference.py` compares a
produced run against stored reference outputs (metrics to 4 decimals,
arrays bitwise, SHA-256 of the npz file).

Reference results on the documented inputs (448 x 304 x 540 monthly
CDR record 1979-01..2023-12, 72,485 ocean cells; targets 2016-01..
2023-12), all reproduced **bitwise** (identical predictions.npz
SHA-256) from a clean environment:

| run | MAE | RMSE | IIEE (10^6 km^2) | ACC |
|---|---|---|---|---|
| arm a, full base | 0.025759 | 0.090791 | 1.198854 | 0.651443 |
| arm b, full base | 0.026268 | 0.089500 | 1.292196 | 0.660041 |
| arm b, 10-year base | 0.025114 | 0.088752 | 1.171128 | 0.665064 |
| seasonal 12/24/36 (= paper's "interannual") | 0.027123 | 0.092128 | 1.331038 | 0.691280 |

Unit tests (46) cover the calendar/anchor map, freshness weights, the
same-month pool, causal climatology and PCA, trend slopes, the
transfers, the metrics on hand-checked arrays, and the CDR conversion
recipes; `pytest` runs them on synthetic data only.

## License

MIT (see `LICENSE`).
