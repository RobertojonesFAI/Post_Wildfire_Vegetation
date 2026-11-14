"""Score the two baselines (persistence and gradient boosting) on the validation fires.

Uses the same validation samples as src/recovery/train.py (same seed and --max-val), so the
numbers can go in one table with the Prithvi runs.

  python scripts/run_baselines.py --dataset $DATASET --out $PROJ/runs/baselines --max-val 2000
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovery.baselines import fit_gbm  # noqa: E402
from recovery.data import RecoveryDataset, make_index  # noqa: E402
from recovery.train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Persistence and gradient boosting baselines")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--gbm-samples", type=int, default=3000, help="training samples used to fit the GBM")
    ap.add_argument("--pixels", type=int, default=300, help="pixels per sample used to fit the GBM")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=497)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    index = make_index(a.dataset, num_frames=a.frames, horizon=a.horizon)
    counts = index.groupby(["split_fire", "veg_type"]).agg(samples=("chip_id", "size"), chips=("chip_id", "nunique"),
                                                         fires=("mtbs_id", "nunique")).reset_index()
    counts.to_csv(a.out / "sample_counts.csv", index=False)
    print(counts.to_string(index=False))

    tr = index[index["split_fire"] == "train"].sample(frac=1, random_state=a.seed)
    va = index[index["split_fire"] == a.split].sample(frac=1, random_state=a.seed).head(a.max_val)
    print(f"GBM fit on up to {a.gbm_samples} train samples; scoring {len(va)} {a.split} samples")

    gbm = fit_gbm(RecoveryDataset(a.dataset, tr), pixels_per_sample=a.pixels, max_samples=a.gbm_samples, seed=a.seed)
    with open(a.out / "gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)

    loader = DataLoader(RecoveryDataset(a.dataset, va), batch_size=8, num_workers=a.workers)
    table = evaluate(None, loader, torch.device("cpu"), gbm=gbm)
    table.to_csv(a.out / f"baselines_{a.split}.csv", index=False)
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))

    nbr = table[~table["method"].str.contains("recovery")]
    print(nbr.pivot(index="method", columns="group", values="rmse").round(4).to_string())


if __name__ == "__main__":
    main()
