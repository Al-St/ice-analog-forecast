# Data formats

Everything the model reads, with shapes, dtypes and provenance. No data
files live in this repository (only a 16 x 16 mask crop used by a unit
test); `scripts/prepare_inputs.py` regenerates the arrays from the NSIDC
CDR monthly file.

## Monthly concentrations `sic_monthly.npy`

- shape `(448, 304, 540)`, float64, month axis LAST;
- months 1979-01 .. 2023-12 (540), CDR grid (polar stereographic 25 km,
  NSIDC northern hemisphere);
- values in [0, 1]; flagged CDR values contribute 0.0;
- provenance: `cdr_seaice_conc_monthly` (uint8 percent + flags) of the
  NSIDC CDR monthly file, converted byte-exactly: read with netCDF4 auto
  mask/scale DISABLED; window = the last 542-2 = 540 months (the file
  starts 1978-11, the record starts at file index 2); raw > 100 -> 0.0;
  `float32(raw) * float32(0.01)`; cast float64; month axis last.

## Ocean mask `ocean_mask.npy`

- shape `(448, 304)`, bool, True = ocean;
- 72,485 ocean cells of 136,192;
- provenance: NOT (CDR land flag, raw == 254) read from the LAST month
  of the record. The land-flag set is not perfectly static — three
  months of the record flag parts of the land as missing instead of
  land (481 cells; verified over all 542 months) — and the last month's
  set is the superset covering every other month.

## Latitude `latitude.npy`

- shape `(448, 304)`, float64; the CDR file's `latitude` variable
  (not used by the model core; kept for coordinate checks and plots).

## Region raster `regions_nsidc0780.nc`

The NSIDC-0780 Sea Ice Regions file (polar stereographic 25 km grid,
identical 448 x 304 cell layout). Variables:

- `sea_ice_region`: uint8 raster, flag values 0..18 (flag meanings are
  region names; 0 is "ocean without a specified region");
- `sea_ice_region_surface_mask`: uint8; values < 30 are ocean.

A model region cell is `raster == rid` (rid 1..18) AND raster-file ocean
AND concentration-product ocean. Empty regions are dropped (none are
empty on this grid); the 18 non-empty regions cover 38,521 + unassigned
ocean (id 0) covers 28,871 of the 72,485 ocean cells — hence the
pan-Arctic series of the scalar arm is the SUM OF THE 18 REGIONAL
EXTENTS, not the total-ocean extent.

## Regional extents (model-internal)

`monthly_sie` convention: per region and month, the number of ocean
region cells with concentration > 0.15 times 625 km^2 per cell; float64
`(540, 18)`.

## Predictions `predictions.npz`

Written by `scripts/run_forecast.py` (key order canonical):

| key | shape / dtype | meaning |
|---|---|---|
| `preds` | (rows, 448, 304, leads) float32 | predictions, [0, 1] |
| `truths` | same | verifying concentrations at the targets |
| `target_ym` | `<U7` (`YYYY-MM`) | target month of each row (seasonal scale: per lead) |
| `issue_ym` | `<U7` | issuing month of each row (seasonal scale: per lead) |
| `lead_months` | (leads,) int32 | lead lengths of the last axis |

Scale 1: rows = targets 2016-01..2023-12 (96), leads 1..6.
Seasonal scale: rows = issues 2016-01..2020-12 (60), leads 12/24/36;
`target_ym` is (60, 3).

## Metric summary `metric_summary.json`

`mae_mean`, `rmse_mean`, `iiee_mean`, `acc_mean` (rounded to 6
decimals) over ocean cells; IIEE uses the 0.15 edge threshold and 625
km^2 cells (reported in 10^6 km^2); ACC is the per-cell anomaly
correlation over rows, averaged over ocean cells and leads; plus
`npz_sha256`.
