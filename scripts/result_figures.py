"""Results table and figures once a model is trained.

  python scripts/result_figures.py --dataset $DATASET --run $PROJ/runs/frozen_300m \
      --baselines $PROJ/runs/baselines --out $PROJ/figures

Writes:
  results_table.tex  RMSE of NBR 12 months ahead and skill vs persistence, by vegetation type
  learning_curve     training loss and validation RMSE per epoch
  examples           three validation chips: NBR now, truth, gradient boosting, Prithvi, Prithvi error
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovery.baselines import predict_gbm  # noqa: E402
from recovery.data import RecoveryDataset, make_index  # noqa: E402
from recovery.model import PrithviRecovery  # noqa: E402
from recovery.train import to_device  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight"})
GROUPS = ["all", "forest", "shrubland", "grassland"]
NAMES = {"persistence": "Persistence", "gbm": "Gradient boosting", "prithvi": "Prithvi-EO-2.0 (ours)"}


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def results_table(run, baselines, out, split):
    p = pd.read_csv(run / "val_metrics.csv")
    b = pd.read_csv(baselines / f"baselines_{split}.csv")
    t = pd.concat([b[b["method"].isin(["persistence", "gbm"])], p[p["method"] == "prithvi"]])
    rec = pd.concat([b[b["method"].isin(["persistence (recovery %)", "gbm (recovery %)"])],
                     p[p["method"] == "prithvi (recovery %)"]])
    rec["method"] = rec["method"].str.replace(" (recovery %)", "", regex=False)
    t.to_csv(out / "results_nbr.csv", index=False)
    rec.to_csv(out / "results_recovery.csv", index=False)

    groups = [g for g in GROUPS if g in set(t["group"])]
    lines = [r"\begin{tabular}{l" + "c" * len(groups) + "c}", r"\toprule",
             "Method & " + " & ".join(g.capitalize() if g != "all" else "All" for g in groups)
             + r" & Rec.\ MAE \\", r"\midrule"]
    best = {g: t[t["group"] == g]["rmse"].min() for g in groups}
    for m in ["persistence", "gbm", "prithvi"]:
        cells = []
        for g in groups:
            r = t[(t["method"] == m) & (t["group"] == g)]
            if r.empty:
                cells.append("--")
                continue
            v, s = r["rmse"].iloc[0], r["skill_vs_persistence"].iloc[0]
            txt = f"{v:.3f}" + (f" ({s:+.2f})" if m != "persistence" else "")
            cells.append(r"\textbf{" + txt + "}" if np.isclose(v, best[g]) else txt)
        mae = rec[(rec["method"] == m) & (rec["group"] == "all")]["mae"]
        cells.append(f"{mae.iloc[0]:.1f}" if len(mae) else "--")
        lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", out / "results_table.tex")
    print(t.pivot(index="method", columns="group", values="rmse").round(4).to_string())


def learning_curve(run, out):
    h = pd.read_csv(run / "history.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(h["epoch"], np.sqrt(h["train_mse"]), "o-", ms=3, label="train RMSE")
    ax.plot(h["epoch"], h["val_rmse"], "s-", ms=3, label="val RMSE (Prithvi)")
    ax.plot(h["epoch"], h["val_rmse_persistence"], "--", color="0.5", label="val RMSE (persistence)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NBR RMSE")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, out, "learning_curve")


def examples(dataset, run, baselines, out, n=3):
    cfg = json.loads((run / "config.json").read_text())
    index = make_index(dataset, num_frames=cfg["num_frames"], horizon=cfg["horizon"])
    va = index[index["split_fire"] == cfg["val_split"]].sample(frac=1, random_state=cfg["seed"]).head(cfg["max_val"])
    ds = RecoveryDataset(dataset, va)
    # one sample per vegetation type, the one with the most valid burned pixels
    picks = []
    for veg in ["forest", "shrubland", "grassland"]:
        cand = [i for i in range(len(ds)) if va.iloc[i]["veg_type"] == veg][:200]
        if cand:
            picks.append(max(cand, key=lambda i: int(ds[i]["mask"].sum())))
    picks = picks[:n]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrithviRecovery(cfg["backbone"], cfg["num_frames"], cfg["freeze_encoder"], cfg["use_terrain"])
    model.load_state_dict(torch.load(run / "best_head.pt", map_location="cpu", weights_only=True), strict=False)
    model.to(device).eval()
    gbm = pickle.load(open(baselines / "gbm.pkl", "rb"))

    fig, axes = plt.subplots(len(picks), 5, figsize=(6.9, 1.45 * len(picks)), squeeze=False)
    fig.subplots_adjust(wspace=0.45)
    for n_row, (row, i) in enumerate(zip(axes, picks)):
        item = ds[i]
        with torch.no_grad():
            pred = model(to_device({k: v.unsqueeze(0) for k, v in item.items()}, device))[0].float().cpu().numpy()
        mask = item["mask"].numpy()
        show = lambda a: np.where(mask, a, np.nan)  # noqa: E731
        truth = item["target"].numpy()
        panels = [("NBR now", item["base"].numpy()), (f"truth (+{cfg['horizon']} mo)", truth),
                  ("gradient boosting", predict_gbm(gbm, item)), ("Prithvi (ours)", pred),
                  ("Prithvi - truth", pred - truth)]
        r = va.iloc[i]
        for k, (ax, (title, a)) in enumerate(zip(row, panels)):
            err = k == 4
            im = ax.imshow(show(a), cmap="RdBu" if err else "RdYlGn", vmin=-0.3 if err else -0.2,
                           vmax=0.3 if err else 0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if n_row == 0:
                ax.set_title(title)
            if k in (3, 4):
                cax = ax.inset_axes([1.03, 0.05, 0.05, 0.9])
                fig.colorbar(im, cax=cax).ax.tick_params(labelsize=5)
        row[0].set_ylabel(f"{str(r['name']).title()}\n{r['veg_type']}\n{r['month']}", fontsize=6.5)
    save(fig, out, "examples")


def main():
    ap = argparse.ArgumentParser(description="Results table and figures")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, type=Path, help="output folder of src/recovery/train.py")
    ap.add_argument("--baselines", required=True, type=Path, help="output folder of scripts/run_baselines.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results_table(a.run, a.baselines, a.out, a.split)
    learning_curve(a.run, a.out)
    examples(a.dataset, a.run, a.baselines, a.out)


if __name__ == "__main__":
    main()
