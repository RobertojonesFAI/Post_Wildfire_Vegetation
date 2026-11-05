"""Training samples from the dataset made by scripts/build_dataset.py.

A sample is one chip at one month t:
  input  = the last `num_frames` mostly-clear months up to t (all after the fire), 6 HLS bands each
  target = NBR at month t + horizon, on the burned pixels that are clear that month
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NODATA = -9999
CHIP_M = 224 * 30
# Prithvi-EO-2.0 normalization for HLS reflectance x 10,000 (BLUE, GREEN, RED, NIR, SWIR1, SWIR2)
MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)


def make_index(dataset_dir, num_frames=4, horizon=12, min_clear=0.5, min_target_clear=0.3, max_span=12):
    """One row per sample: chip, input month positions, target month position."""
    d = Path(dataset_dir)
    chips = pd.read_csv(d / "chips.csv")
    chips = chips[chips["status"] == "ok"]
    months = pd.read_csv(d / "chip_months.csv")
    months = months[months["chip_id"].isin(chips["chip_id"])]
    rows = []
    for chip_id, g in months.groupby("chip_id"):
        g = g.sort_values("month").reset_index(drop=True)
        clear = g["clear_frac"].to_numpy()
        burned_clear = g["burned_clear_frac"].to_numpy()
        since = g["months_since_fire"].to_numpy()
        good = np.flatnonzero((clear >= min_clear) & (since >= 1))
        for k in range(num_frames - 1, len(good)):
            frames = good[k - num_frames + 1:k + 1]
            t, target = int(frames[-1]), int(frames[-1]) + horizon
            if frames[-1] - frames[0] > max_span or target >= len(g) or burned_clear[target] < min_target_clear:
                continue
            rows.append({"chip_id": chip_id, "frames": [int(f) for f in frames], "t": t, "target": target,
                         "month": g.at[t, "month"], "target_month": g.at[target, "month"],
                         "months_since_fire": int(since[t])})
    cols = ["chip_id", "mtbs_id", "name", "region", "veg_type", "split_fire", "epsg", "x", "y"]
    return pd.DataFrame(rows).merge(chips[cols], on="chip_id")


def chip_latlon(epsg, x, y):
    """Center of a chip in degrees."""
    from rasterio.warp import transform
    lon, lat = transform(epsg, "EPSG:4326", [x + CHIP_M / 2], [y - CHIP_M / 2])
    return float(lat[0]), float(lon[0])


def terrain(static):
    """Elevation (km), slope (/45 deg), aspect as sin/cos -> (4, H, W)."""
    elev = static[3] / 1000.0
    slope = static[4] / 100.0 / 45.0
    aspect = np.deg2rad(static[5])
    return np.stack([elev, slope, np.sin(aspect), np.cos(aspect)]).astype(np.float32)


class ChipCache:
    """Open (memory-mapped) chips, most recently used first."""
    KEYS = ("refl", "valid", "nbr", "burned", "nbr_pre", "nbr_post", "static", "months")

    def __init__(self, dataset_dir, size=64):
        self.dir, self.size, self.items = Path(dataset_dir), size, OrderedDict()

    def get(self, chip_id):
        if chip_id in self.items:
            self.items.move_to_end(chip_id)
            return self.items[chip_id]
        # memory-mapped: only the months a sample uses are read from disk
        item = {k: np.load(self.dir / chip_id / f"{k}.npy", mmap_mode="r") for k in self.KEYS}
        self.items[chip_id] = item
        if len(self.items) > self.size:
            self.items.popitem(last=False)
        return item


def _augment(arrays, rng):
    k, flip = rng.integers(4), rng.random() < 0.5
    out = []
    for a in arrays:
        a = np.rot90(a, k, axes=(-2, -1))
        out.append(np.ascontiguousarray(a[..., ::-1] if flip else a))
    return out


class RecoveryDataset(Dataset):
    def __init__(self, dataset_dir, index, augment=False, cache_size=64, seed=0):
        self.index = index.reset_index(drop=True)
        self.cache = ChipCache(dataset_dir, cache_size)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        first = self.index.drop_duplicates("chip_id")
        self.latlon = {r.chip_id: chip_latlon(r.epsg, r.x, r.y) for r in first.itertuples()}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index.iloc[i]
        c = self.cache.get(r.chip_id)
        frames = list(r.frames)

        x = c["refl"][frames].astype(np.float32)                  # (T, 6, H, W)
        missing = (x == NODATA) | ~c["valid"][frames][:, None]
        x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
        x[missing] = 0.0

        nbr_in = c["nbr"][frames].astype(np.float32)               # (T, H, W), NaN = no data
        base = np.full(nbr_in.shape[1:], np.nan, dtype=np.float32)
        for f in nbr_in:                                           # last clear value of each pixel
            base = np.where(np.isfinite(f), f, base)
        target = c["nbr"][r.target].astype(np.float32)
        mask = c["burned"] & np.isfinite(target) & np.isfinite(base)

        maps = [x, nbr_in, base, target, mask, np.asarray(c["nbr_pre"]), np.asarray(c["nbr_post"]),
                terrain(np.asarray(c["static"], dtype=np.float32)), c["static"][2].astype(np.float32)]
        if self.augment:
            maps = _augment(maps, self.rng)
        x, nbr_in, base, target, mask, pre, post, ter, nlcd = maps

        dates = [pd.Timestamp(f"{c['months'][f]}-15") for f in frames]
        target_month = int(str(c["months"][r.target])[5:7])
        return {
            "x": torch.from_numpy(x).permute(1, 0, 2, 3),          # (6, T, H, W) as Prithvi wants
            "temporal_coords": torch.tensor([[d.year, d.dayofyear - 1] for d in dates], dtype=torch.float32),
            "location_coords": torch.tensor(self.latlon[r.chip_id], dtype=torch.float32),
            "base": torch.from_numpy(np.nan_to_num(base)),
            "target": torch.from_numpy(np.nan_to_num(target)),
            "mask": torch.from_numpy(mask),
            "nbr_in": torch.from_numpy(nbr_in),
            "nbr_pre": torch.from_numpy(np.nan_to_num(pre)),
            "nbr_post": torch.from_numpy(np.nan_to_num(post)),
            "terrain": torch.from_numpy(ter),
            "nlcd": torch.from_numpy(nlcd),
            "months_since_fire": torch.tensor(r.months_since_fire, dtype=torch.float32),
            "target_month": torch.tensor(target_month, dtype=torch.float32),
            "i": torch.tensor(i),
        }
