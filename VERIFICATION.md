# Verification record

Environment: Python 3.10.11, Windows x64, clean virtual environment from
`requirements.txt` (numpy 1.26.4, scipy 1.12.0, netCDF4 1.7.4,
xarray 2025.6.1, pytest 8.3.4). Inputs regenerated from the NSIDC CDR
monthly file by `scripts/prepare_inputs.py` and checked bitwise against
previously prepared work arrays (SIC: bitwise equal; ocean mask: bitwise
equal, 72,485 ocean cells).

Commands (from the repository root):

```
python scripts/run_forecast.py --work <data dir> --arm a --base-years 0 \
    --out outputs/a_full
python scripts/verify_reference.py --produced outputs/a_full \
    --reference <reference dir> --ocean <data dir>/ocean_mask.npy \
    --require-bitwise
```

All four runs reproduce the stored reference outputs **bitwise**: every
npz array (`preds`, `truths`, `target_ym`, `issue_ym`, `lead_months`)
is `array_equal` to the reference, and the SHA-256 of the npz files is
identical. Metrics therefore agree to all 6 documented decimals (the
reference summaries store 6 decimals; 4-decimal agreement follows).

| run | reference SHA-256 (first 16) | produced SHA-256 (first 16) | MAE | RMSE | IIEE (10^6 km^2) | ACC | verdict |
|---|---|---|---|---|---|---|---|
| arm a, full base | `6ab559607d3a46a3` | `6ab559607d3a46a3` | 0.025759 | 0.090791 | 1.198854 | 0.651443 | bitwise PASS |
| arm b, full base | `9b458e969650d229` | `9b458e969650d229` | 0.026268 | 0.089500 | 1.292196 | 0.660041 | bitwise PASS |
| arm b, 10-year base | `148570bd2809ab80` | `148570bd2809ab80` | 0.025114 | 0.088752 | 1.171128 | 0.665064 | bitwise PASS |
| seasonal 12/24/36 | `25a59263ddea5f0a` | `25a59263ddea5f0a` | 0.027123 | 0.092128 | 1.331038 | 0.691280 | bitwise PASS |

Bitwise reproduction of the regional arm means the PCA-5 decomposition
(full SVD with the largest-|score| sign convention) matches the
deterministic reference decomposition exactly on this platform with the
pinned scipy build; do not bump numpy/scipy casually (see README).

Unit tests: 46 passed (`pytest`), synthetic data only (the only real
data in the repository is a 16 x 16 ocean-mask fixture crop, pinned to
256 ocean cells).

## Post-review update: r03-C08 terminology mapping

After review round 3 (comment r03-C08: the manuscript calls the
12/24/36-month scheme "the interannual scale" while the CLI spells it
`seasonal`), the CLI additionally accepts `--scale interannual` as an
exact alias, canonicalized to `seasonal` (`metadata.json` and outputs
are byte-identical under either spelling; see README "Terminology").
Re-verified after the change, from the same clean environment:

```
python scripts/run_forecast.py --work <data dir> --arm b \
    --scale interannual --out outputs/seasonal_alias_c08
python scripts/verify_reference.py --produced outputs/seasonal_alias_c08 \
    --reference <reference dir> --ocean <data dir>/ocean_mask.npy \
    --require-bitwise
python scripts/run_forecast.py --work <data dir> --arm a --base-years 0 \
    --out outputs/a_full_c08_check
python scripts/verify_reference.py --produced outputs/a_full_c08_check \
    --reference <arm-a reference dir> --ocean <data dir>/ocean_mask.npy \
    --require-bitwise
```

| run | reference SHA-256 (first 16) | produced SHA-256 (first 16) | verdict |
|---|---|---|---|
| seasonal 12/24/36, via `--scale interannual` | `25a59263ddea5f0a` | `25a59263ddea5f0a` | bitwise PASS |
| arm a, full base (re-check) | `6ab559607d3a46a3` | `6ab559607d3a46a3` | bitwise PASS |

Unit tests after the change: 49 passed (46 + 3 alias-mapping tests in
`tests/test_run_forecast_cli.py`).
