"""CDR conversion recipes (concentration, land flag) and the real-mask
fixture fragment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analog_ice.inputs import monthly_sic_from_cdr

FIXTURES = Path(__file__).parent / "fixtures"


def _write_synthetic_cdr(path: Path, raw: np.ndarray) -> None:
    import netCDF4
    t, y, x = raw.shape
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("tdim", t)
    ds.createDimension("y", y)
    ds.createDimension("x", x)
    var = ds.createVariable("cdr_seaice_conc_monthly", "u1",
                            ("tdim", "y", "x"))
    var[:] = raw
    lat = ds.createVariable("latitude", "f4", ("y", "x"))
    lat[:] = np.arange(y * x, dtype=np.float32).reshape(y, x)
    ds.close()


def test_cdr_conversion_rules(tmp_path):
    """Flagged values (>100) -> 0.0; valid values rescale as
    float32(raw) * float32(0.01); months are the LAST n_months slices
    (the real product carries two leading months before 1979-01)."""
    raw = np.zeros((4, 3, 2), dtype=np.uint8)
    raw[2, 0, 0] = 100       # 100 * 0.01 = 1.00 (valid, not a flag)
    raw[2, 1, 0] = 37        # 0.37
    raw[2, 2, 0] = 60        # 0.60
    raw[3, 0, 0] = 255       # missing flag -> 0.0
    raw[3, 1, 0] = 253       # coastal flag -> 0.0
    raw[3, 2, 0] = 254       # land flag -> 0.0, and marks land
    path = tmp_path / "synthetic_cdr.nc"
    _write_synthetic_cdr(path, raw)

    sic, ocean = monthly_sic_from_cdr(path, start_year=1979, n_months=2)
    # the same arithmetic as the recipe: float32(raw) * float32(0.01)
    expected = np.zeros((3, 2, 2), dtype=np.float64)
    expected[0, 0, 0] = float(np.float32(100) * np.float32(0.01))
    expected[1, 0, 0] = float(np.float32(37) * np.float32(0.01))
    expected[2, 0, 0] = float(np.float32(60) * np.float32(0.01))
    assert np.array_equal(sic, expected)
    assert sic.dtype == np.float64
    # land from the LAST window slice (index 3): only (2, 0)
    assert ocean.dtype == np.bool_
    assert ocean[2, 0] == False  # noqa: E712  (land cell)
    assert ocean[0, 0] and ocean[1, 0]


def test_land_flag_is_static_across_slices(tmp_path):
    """The recipe reads land from the last slice; with the flag present in
    every slice the mask is identical for any window choice."""
    raw = np.full((5, 2, 2), 30, dtype=np.uint8)
    raw[:, 0, 1] = 254
    path = tmp_path / "static_land.nc"
    _write_synthetic_cdr(path, raw)
    _, ocean = monthly_sic_from_cdr(path, start_year=1979, n_months=3)
    assert ocean[0, 1] == False  # noqa: E712
    assert ocean[0, 0]


def test_real_mask_fixture_fragment() -> None:
    """A 16 x 16 crop of the real ocean mask (grid top-left corner) is
    pinned in the repo. It was produced by prepare_inputs.py from the CDR
    file (land = flag 254); the full mask has 72,485 ocean cells of
    136,192 (documented in data_format.md)."""
    fragment = np.load(FIXTURES / "ocean_mask_fragment_16x16.npy")
    assert fragment.dtype == np.bool_ and fragment.shape == (16, 16)
    # pinned pattern of this crop (grid top-left corner is open ocean:
    # 256 of 256 cells), verified against the CDR land flag when the
    # fixture was cut
    assert int(fragment.sum()) == 256
