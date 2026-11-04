"""Regional predictor series: monthly sea-ice extent of the 18 NSIDC
sea-ice regions and the pan-Arctic series derived from them.

Region loading reads the NSIDC-0780 region raster (polar stereographic,
25 km, same 448 x 304 grid as the concentration product) with netCDF4
auto mask/scale disabled. The convention used by the model:

- ``sea_ice_region`` is the region raster (flag values 0..18; 0 is
  "ocean without a specified region" and is NOT a region);
- ``sea_ice_region_surface_mask`` distinguishes ocean (values < 30) from
  the non-ocean surface inside the region file;
- a region cell for the model is: raster == region id AND region-file
  ocean AND concentration-product ocean (the CDR land flag).

Monthly regional extent follows the ``monthly_sie`` convention: count of
region ocean cells with SIC > 0.15 times 625 km2 per cell. The
pan-Arctic series is the SUM of the 18 regional extents (regions cover
the peripheral seas plus the central Arctic; unassigned ocean id 0 is
excluded by design).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from netCDF4 import Dataset

CELL_AREA_KM2 = 625.0
SIC_THRESHOLD = 0.15
REGION_VARIABLE = "sea_ice_region"
SURFACE_VARIABLE = "sea_ice_region_surface_mask"
SURFACE_OCEAN_MAX = 29  # surface values < 30 are ocean
UNASSIGNED_ID = 0

REGION_NAMES = {
    1: "central_arctic",
    2: "beaufort_sea",
    3: "chukchi_sea",
    4: "east_siberian_sea",
    5: "laptev_sea",
    6: "kara_sea",
    7: "barents_sea",
    8: "east_greenland_sea",
    9: "baffin_bay_and_labrador_seas",
    10: "gulf_of_st_lawrence",
    11: "hudson_bay",
    12: "canadian_archipelago",
    13: "bering_sea",
    14: "sea_of_okhotsk",
    15: "sea_of_japan",
    16: "bohai_and_yellow_seas",
    17: "baltic_sea",
    18: "gulf_of_alaska",
}


@dataclass(frozen=True)
class RegionMasks:
    """The 18 boolean region masks intersected with the ocean mask.

    Attributes:
        ids: region ids, ascending (1..18, empty masks dropped).
        names: region id -> name.
        masks: region id -> bool (H, W) mask (ocean cells of the region).
    """

    ids: list[int]
    names: dict[int, str]
    masks: dict[int, np.ndarray]

    @property
    def stack(self) -> list[np.ndarray]:
        """Masks as a list aligned with :attr:`ids`."""
        return [self.masks[rid] for rid in self.ids]


def load_region_masks(nc_path: Path, ocean: np.ndarray) -> RegionMasks:
    """Load the 18 NSIDC region masks on the model grid.

    Reads the region raster with netCDF4 auto mask/scale disabled,
    intersects every region (ids 1..18, raster flag < 30; the unassigned
    id 0 is skipped) with the region-file ocean surface and the
    concentration-product ocean mask, and drops empty masks.
    """
    with Dataset(str(nc_path)) as ds:
        ds.set_auto_maskandscale(False)
        if REGION_VARIABLE not in ds.variables:
            raise ValueError(f"{nc_path}: missing {REGION_VARIABLE}")
        reg = np.array(ds.variables[REGION_VARIABLE][:]).astype(np.int32)
        surface_var = SURFACE_VARIABLE if SURFACE_VARIABLE in ds.variables \
            else REGION_VARIABLE
        surface = np.array(ds.variables[surface_var][:]).astype(np.int32)

    region_ocean = surface < SURFACE_OCEAN_MAX + 1
    if reg.shape != ocean.shape or region_ocean.shape != ocean.shape:
        raise ValueError(
            f"region grid {reg.shape} does not match the ocean mask {ocean.shape}")

    ids: list[int] = []
    names: dict[int, str] = {}
    masks: dict[int, np.ndarray] = {}
    for rid in sorted(REGION_NAMES):
        mask = (reg == rid) & region_ocean & ocean
        if mask.sum() == 0:
            continue
        ids.append(rid)
        names[rid] = REGION_NAMES[rid]
        masks[rid] = mask
    if len(ids) < 2:
        raise ValueError(f"{nc_path}: fewer than two non-empty regions found")
    return RegionMasks(ids=ids, names=names, masks=masks)


def regional_extents(sic: np.ndarray, region_masks: RegionMasks) -> np.ndarray:
    """Monthly regional extents, float64 (N, R) in km2.

    ``monthly_sie`` convention: per region and month, the number of ocean
    cells of the region with SIC > 0.15 times 625 km2.
    """
    masks = region_masks.stack
    exceeded = sic > SIC_THRESHOLD  # (H, W, N)
    out = np.empty((sic.shape[2], len(masks)), dtype=np.float64)
    for j, mask in enumerate(masks):
        counts = exceeded[mask, :].sum(axis=0)
        out[:, j] = counts * CELL_AREA_KM2
    return out


def pan_arctic_extent(sic: np.ndarray, region_masks: RegionMasks) -> np.ndarray:
    """Pan-Arctic monthly extent series, float64 (N,) in km2: the sum of
    the 18 regional extents (the model's scalar predictor)."""
    return regional_extents(sic, region_masks).sum(axis=1)
