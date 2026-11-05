"""Turn the HLS chips from notebook 02 into training-ready files.

For every chip it writes one folder of .npy arrays with the monthly bands and the recovery
targets (uncompressed, so training can read just the months it needs), plus three tables:

  chips.csv        one row per chip: fire, region, vegetation, split, burn and recovery summary
  chip_months.csv  one row per chip and month: clear share, mean NBR / recovery, weather
  splits.json      how train / val / test are defined

Usage (Borah or Colab):
  python scripts/build_dataset.py --chips /path/to/cs497_hls_chips --out /path/to/dataset

What goes in each chip folder (T = number of months, H = W = 224), one <name>.npy per array:
  refl      (T, 6, H, W) int16    BLUE, GREEN, RED, NIR, SWIR1, SWIR2, reflectance x 10,000, -9999 = no data
  n         (T, H, W)    uint8    clear looks in the monthly composite
  valid     (T, H, W)    bool     the pixel has a clear value that month
  static    (6, H, W)    int16    PERIM, SEVERITY, NLCD, ELEV, SLOPE_X100, ASPECT
  nbr, ndvi (T, H, W)    float16  NaN where not valid
  nbr_pre   (H, W)       float32  mean summer NBR in the 2 years before the fire
  nbr_post  (H, W)       float32  lowest NBR in the 4 months from ignition
  dnbr      (H, W)       float32  nbr_pre - nbr_post
  burned    (H, W)       bool     inside the perimeter and dNBR >= 0.1
  rec       (T, H, W)    float16  recovery % = (NBR_t - NBR_post) / (NBR_pre - NBR_post) * 100,
                                  only for burned pixels after ignition, NaN elsewhere
  ytr       (H, W)       float16  years to recovery: first summer with NBR >= 80% of NBR_pre,
                                  counted from the fire year; NaN = not recovered yet
  months    (T,)         str      "YYYY-MM"
  ignition  ()           str      "YYYY-MM"
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parents[1]

N_STATIC = 6
N_PER_MONTH = 7  # 6 bands + N
NODATA = -9999
SEASON = (6, 7, 8, 9)   # June-September, so we always compare the same season
POST_WINDOW = 4         # months from ignition used for NBR right after the fire
MIN_DNBR = 0.1          # a pixel counts as burned if its NBR dropped at least this much
RECOVERED = 0.8         # recovered = summer NBR back to 80% of the pre-fire mean
TIME_SPLIT = {"train": "until 2023-12", "val": "2024-01 to 2024-12", "test": "2025-01 onward"}
SEED = 497

# Wapiti is not in MTBS, so it is not in fires.csv with a date. Start date from InciWeb.
EXTRA_IGNITIONS = {"WAPITI2024": "2024-07-24"}


def read_chip(chip_dir, chip_id):
    """Static bands and monthly bands of a chip, from one GeoTIFF or from its _p0, _p1... parts."""
    single = chip_dir / f"{chip_id}.tif"
    if single.exists():
        files = [single]
    else:
        files = sorted(chip_dir.glob(f"{chip_id}_p*.tif"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
    if not files:
        return None, None
    static, monthly = None, []
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read()
        if static is None:
            static = arr[:N_STATIC]
        monthly.append(arr[N_STATIC:])
    return static, np.concatenate(monthly)


def normalized_difference(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def nan_reduce(fn, stack):
    """nanmean / nanmin over the first axis without the all-NaN warnings."""
    if len(stack) == 0:
        return np.full(stack.shape[1:], np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(stack, axis=0).astype(np.float32)


def nan_mean(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values)) if values.size else np.nan


def process_chip(job):
    chip, ignition, chip_dir, out_dir = job
    static, monthly = read_chip(Path(chip_dir), chip["chip_id"])
    if static is None:
        return {"chip_id": chip["chip_id"], "status": "missing"}, []

    months = pd.period_range(chip["start_month"], chip["end_month"], freq="M")
    T, H, W = len(months), static.shape[1], static.shape[2]
    if monthly.shape[0] != T * N_PER_MONTH:
        return {"chip_id": chip["chip_id"], "status": f"expected {T * N_PER_MONTH} monthly bands, got {monthly.shape[0]}"}, []

    monthly = monthly.reshape(T, N_PER_MONTH, H, W)
    refl, n = monthly[:, :6], monthly[:, 6]
    valid = (n > 0) & (refl != NODATA).all(axis=1)

    nir = np.where(valid, refl[:, 3], np.nan).astype(np.float32)
    red = np.where(valid, refl[:, 2], np.nan).astype(np.float32)
    swir2 = np.where(valid, refl[:, 5], np.nan).astype(np.float32)
    nbr = normalized_difference(nir, swir2)
    ndvi = normalized_difference(nir, red)

    ig = pd.Period(ignition, freq="M")
    month_of_year = np.array([p.month for p in months])
    year = np.array([p.year for p in months])
    summer = np.isin(month_of_year, SEASON)
    before = np.array([p < ig for p in months])
    after = np.array([p >= ig for p in months])
    post_window = np.array([ig <= p < ig + POST_WINDOW for p in months])

    nbr_pre = nan_reduce(np.nanmean, nbr[before & summer])
    nbr_post = nan_reduce(np.nanmin, nbr[post_window])
    dnbr = nbr_pre - nbr_post
    inside = static[0] == 1
    with np.errstate(invalid="ignore"):
        burned = inside & (dnbr >= MIN_DNBR)

    with np.errstate(divide="ignore", invalid="ignore"):
        rec = (nbr - nbr_post) / (nbr_pre - nbr_post) * 100
    rec[:, ~burned] = np.nan
    rec[~after] = np.nan

    # Years to recovery: first summer after the fire year with NBR >= 80% of the pre-fire mean
    ytr = np.full((H, W), np.nan, dtype=np.float32)
    can_recover = burned & (nbr_pre > 0)
    last_month = months[-1]
    observed_years = 0
    for yr in sorted(set(year[year > ig.year])):
        sel = summer & (year == yr)
        if not sel.any():
            continue
        summer_nbr = nan_reduce(np.nanmean, nbr[sel])
        with np.errstate(invalid="ignore"):
            newly = np.isnan(ytr) & can_recover & (summer_nbr >= RECOVERED * nbr_pre)
        ytr[newly] = yr - ig.year
        if pd.Period(f"{yr}-{max(SEASON):02d}", freq="M") <= last_month:
            observed_years = yr - ig.year

    arrays = dict(
        refl=refl.astype(np.int16), n=n.clip(0, 255).astype(np.uint8), valid=valid,
        static=static.astype(np.int16), nbr=nbr.astype(np.float16), ndvi=ndvi.astype(np.float16),
        nbr_pre=nbr_pre, nbr_post=nbr_post, dnbr=dnbr, burned=burned,
        rec=rec.astype(np.float16), ytr=ytr.astype(np.float16),
        months=np.array([str(p) for p in months]), ignition=np.array(str(ig)),
    )
    chip_out = Path(out_dir) / chip["chip_id"]
    chip_out.mkdir(exist_ok=True)
    for name, arr in arrays.items():
        np.save(chip_out / f"{name}.npy", arr)

    unburned = (~inside)[None] & valid
    rows = []
    for t, p in enumerate(months):
        b = burned & valid[t]
        rows.append({
            "chip_id": chip["chip_id"], "month": str(p), "months_since_fire": (p - ig).n,
            "clear_frac": round(float(valid[t].mean()), 4),
            "burned_clear_frac": round(float(b.sum() / max(burned.sum(), 1)), 4),
            "nbr_burned": nan_mean(nbr[t][b]), "ndvi_burned": nan_mean(ndvi[t][b]),
            "rec_burned": nan_mean(rec[t][b]), "nbr_unburned": nan_mean(nbr[t][unburned[t]]),
        })

    recovered = ~np.isnan(ytr[burned])
    summary = {
        "chip_id": chip["chip_id"], "status": "ok", "ignition": str(ig),
        "burned_frac": round(float(burned.mean()), 4),
        "nbr_pre_burned": nan_mean(nbr_pre[burned]), "dnbr_burned": nan_mean(dnbr[burned]),
        "recovered_frac": round(float(recovered.mean()), 4) if burned.any() else np.nan,
        "ytr_median": float(np.median(ytr[burned][recovered])) if recovered.any() else np.nan,
        "observed_years": observed_years,
    }
    return summary, rows


def fire_splits(fires, seed):
    """train / val / test by fire, balanced by vegetation type. The case study is only for testing."""
    rng = np.random.default_rng(seed)
    split = {}
    study = fires[fires["role"] == "study"]
    for _, group in study.groupby("veg_type"):
        ids = sorted(group["mtbs_id"])
        rng.shuffle(ids)
        n_test = max(1, round(0.2 * len(ids)))
        n_val = max(1, round(0.2 * len(ids))) if len(ids) >= 3 else 0
        for i, fid in enumerate(ids):
            split[fid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for fid in fires.loc[fires["role"] != "study", "mtbs_id"]:
        split[fid] = "case_study"
    return split


def load_weather(path):
    w = pd.read_csv(path)
    for col in ("tmmx", "tmmn"):
        w[col] = w[col] - 273.15  # Kelvin -> Celsius
    return w.rename(columns={"pr": "rain_mm", "tmmx": "tmax_c", "tmmn": "tmin_c", "vpd": "vpd_kpa"})[
        ["chip_id", "month", "rain_mm", "tmax_c", "tmin_c", "vpd_kpa"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chips", required=True, type=Path, help="folder with the GeoTIFFs from notebook 02")
    ap.add_argument("--out", required=True, type=Path, help="where to write the dataset")
    ap.add_argument("--manifest", type=Path, default=REPO / "data" / "chips_manifest.csv")
    ap.add_argument("--fires", type=Path, default=REPO / "data" / "fires" / "fires.csv")
    ap.add_argument("--weather", type=Path, default=None,
                    help="gridmet_monthly.csv (default: look for it in --chips)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-fire", type=int, default=None,
                    help="only use this many chips per fire (quick tests)")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    fires = pd.read_csv(args.fires)
    ignitions = {**dict(zip(fires["mtbs_id"], fires["ignition_date"])), **EXTRA_IGNITIONS}
    missing = set(manifest["mtbs_id"]) - set(ignitions)
    if missing:
        raise SystemExit(f"No ignition date for: {sorted(missing)}")

    # Only chips whose GeoTIFFs are there (exports may still be running)
    have = {p.stem.rsplit("_p", 1)[0] if p.stem.rsplit("_p", 1)[-1].isdigit() else p.stem
            for p in args.chips.glob("*.tif")}
    found = manifest[manifest["chip_id"].isin(have)]
    print(f"{len(found)} of {len(manifest)} chips found in {args.chips}")
    if args.max_per_fire:
        found = found.groupby("mtbs_id", group_keys=False).head(args.max_per_fire)
        print(f"Using {len(found)} chips (at most {args.max_per_fire} per fire)")

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(chip, ignitions[chip["mtbs_id"]], str(args.chips), str(args.out))
            for chip in found.to_dict("records")]
    summaries, month_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (summary, rows) in enumerate(pool.map(process_chip, jobs), 1):
            summaries.append(summary)
            month_rows += rows
            print(f"[{i}/{len(jobs)}] {summary['chip_id']}: {summary['status']}", flush=True)

    # Fire splits: Wapiti (and anything not in fires.csv as a study fire) is the case study
    roles = manifest.drop_duplicates("mtbs_id")[["mtbs_id", "veg_type", "role"]]
    split = fire_splits(roles, SEED)

    chips = found.merge(pd.DataFrame(summaries), on="chip_id", how="left")
    chips["split_fire"] = chips["mtbs_id"].map(split)
    chips.to_csv(args.out / "chips.csv", index=False)

    months = pd.DataFrame(month_rows)
    weather_path = args.weather or args.chips / "gridmet_monthly.csv"
    if weather_path.exists() and len(months):
        months = months.merge(load_weather(weather_path), on=["chip_id", "month"], how="left")
    else:
        print("No gridmet_monthly.csv found, chip_months.csv has no weather columns")
    months.to_csv(args.out / "chip_months.csv", index=False)

    (args.out / "splits.json").write_text(json.dumps({
        "fire": split,
        "time": TIME_SPLIT,
        "region": "leave one region out: train on 4 regions, test on the 5th (case study excluded)",
        "seed": SEED,
        "settings": {"season": SEASON, "post_window_months": POST_WINDOW, "min_dnbr": MIN_DNBR,
                     "recovered_share_of_pre": RECOVERED},
    }, indent=2))

    ok = chips[chips["status"] == "ok"]
    print(f"\n{len(ok)} of {len(chips)} chips done -> {args.out}")
    bad = chips[chips["status"] != "ok"]
    if len(bad):
        print(bad[["chip_id", "status"]].to_string(index=False))
    if len(ok):
        print(ok.groupby("split_fire")["mtbs_id"].nunique().rename("fires").to_string())
        print(ok.groupby("veg_type")[["burned_frac", "dnbr_burned", "recovered_frac", "ytr_median"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
