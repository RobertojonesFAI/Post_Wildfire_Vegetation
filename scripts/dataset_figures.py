"""Figures and tables that describe the dataset (no model needed).

  python scripts/dataset_figures.py --dataset $DATASET --out $PROJ/figures

Writes (PDF for the report, PNG to look at):
  trajectories    NBR of burned vs unburned pixels over time, one example fire per vegetation type
  recovery_curves median summer recovery % by years since fire and vegetation type
  chip_example    one chip: true color before / after, pre- and post-fire NBR, dNBR, recovery today
  dataset_table.tex  fires and chips per split and vegetation type (LaTeX)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VEG = ["forest", "shrubland", "grassland"]
COLORS = {"forest": "#1b7837", "shrubland": "#b35806", "grassland": "#c2a000", "mixed": "#762a83"}
SUMMER = ["06", "07", "08", "09"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def pick_chip(chips, months, veg):
    """A clear example: lots of burned area, many observed years and many clear months."""
    c = chips[(chips["veg_type"] == veg) & (chips["status"] == "ok") & (chips["role"] == "study")].copy()
    if c.empty:
        return None
    clear = months[months["clear_frac"] >= 0.5].groupby("chip_id").size()
    c["clear_months"] = c["chip_id"].map(clear).fillna(0)
    c["score"] = c["burned_frac"].fillna(0) + c["observed_years"].fillna(0) / 6 + c["clear_months"] / 120 \
        + 2 * c["dnbr_burned"].fillna(0)
    vegetated = c[c["nbr_pre_burned"] > 0.1]          # a visible drop needs some vegetation before the fire
    return (vegetated if len(vegetated) else c).sort_values("score", ascending=False).iloc[0]


def trajectories(chips, months, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9), sharey=True)
    for ax, veg in zip(axes, VEG):
        chip = pick_chip(chips, months, veg)
        if chip is None:
            ax.set_visible(False)
            continue
        m = months[(months["chip_id"] == chip["chip_id"]) & (months["clear_frac"] >= 0.5)].sort_values("month")
        burned = m[m["burned_clear_frac"] >= 0.5]
        ax.plot(m["months_since_fire"] / 12, m["nbr_unburned"], ".-", color="0.6", ms=2, lw=0.6, label="unburned")
        ax.plot(burned["months_since_fire"] / 12, burned["nbr_burned"], ".-", color=COLORS[veg], ms=2.5, lw=0.8,
                label="burned")
        ax.axvline(0, color="k", lw=0.6)
        if np.isfinite(chip["nbr_pre_burned"]):
            ax.axhline(0.8 * chip["nbr_pre_burned"], color=COLORS[veg], ls="--", lw=0.6, label="80% of pre-fire")
        ax.set_title(f"{veg}\n{str(chip['name']).title()} fire ({int(chip['year'])})", fontsize=7)
        ax.set_xlabel("years since fire")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NBR (median of pixels)")
    handles = [plt.Line2D([], [], color="0.4", lw=1, label="burned pixels (color = vegetation)"),
               plt.Line2D([], [], color="0.6", lw=0.6, marker=".", label="unburned pixels"),
               plt.Line2D([], [], color="0.4", lw=0.6, ls="--", label="80% of pre-fire NBR"),
               plt.Line2D([], [], color="k", lw=0.6, label="ignition")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2))
    save(fig, out, "trajectories")


def recovery_curves(chips, months, out):
    m = months.merge(chips[["chip_id", "veg_type", "ignition"]], on="chip_id")
    m = m[(m["months_since_fire"] >= 0) & m["month"].str[5:7].isin(SUMMER) & (m["burned_clear_frac"] >= 0.3)]
    m["years"] = m["month"].str[:4].astype(int) - m["ignition"].str[:4].astype(int)
    m = m[m["years"] >= 1]                            # the fire-year summer is mostly the fire itself
    per_chip = m.groupby(["veg_type", "chip_id", "years"])["rec_burned"].median().reset_index()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for veg in VEG:
        g = per_chip[per_chip["veg_type"] == veg].groupby("years")["rec_burned"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        n = g.size()
        q = q[n >= 3].loc[:6]
        if q.empty:
            continue
        ax.plot(q.index, q[0.5], "o-", color=COLORS[veg], ms=3, lw=1,
                label=f"{veg} ({chips[chips['veg_type'] == veg]['mtbs_id'].nunique()} fires)")
        ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS[veg], alpha=0.15, lw=0)
    ax.axhline(80, color="k", ls="--", lw=0.6)
    ax.text(1.05, 82, "80% recovered", fontsize=6)
    ax.set_xlabel("summers after the fire year")
    ax.set_ylabel("recovery % (median, IQR)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, out, "recovery_curves")


def rgb(refl):
    """True color from (6, H, W) reflectance x 10,000, 2-98% stretch."""
    img = np.stack([refl[2], refl[1], refl[0]], -1).astype(np.float32)
    img[refl[0] == -9999] = np.nan
    lo, hi = np.nanpercentile(img, 2), np.nanpercentile(img, 98)
    return np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1), 0, 1), nan=1.0)


def chip_example(chips, months, dataset, out):
    chip = pick_chip(chips, months, "forest")
    if chip is None:
        chip = pick_chip(chips, months, chips["veg_type"].dropna().iloc[0])
    d = Path(dataset) / chip["chip_id"]
    arr = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("refl", "valid", "nbr", "nbr_pre", "nbr_post", "dnbr",
                                                               "burned", "rec", "months", "static")}
    mm = pd.Series(arr["months"].astype(str))
    clear = np.asarray(arr["valid"]).mean(axis=(1, 2))
    ig = str(chip["ignition"])
    summer = mm.str[5:7].isin(SUMMER).to_numpy()
    before = np.flatnonzero(summer & (mm < ig).to_numpy() & (clear > 0.9))
    after = np.flatnonzero(summer & (mm > ig).to_numpy() & (clear > 0.9))
    t0 = before[-1] if len(before) else int(np.argmax(clear))
    t1 = after[-1] if len(after) else len(mm) - 1
    burned = np.asarray(arr["burned"])
    perim = np.asarray(arr["static"][0]) == 1
    panels = [
        (f"true color {mm[t0]}", rgb(arr["refl"][t0]), None),
        (f"true color {mm[t1]}", rgb(arr["refl"][t1]), None),
        ("NBR before fire", np.asarray(arr["nbr_pre"]), ("RdYlGn", -0.2, 0.7)),
        ("NBR right after", np.asarray(arr["nbr_post"]), ("RdYlGn", -0.2, 0.7)),
        ("dNBR (severity)", np.asarray(arr["dnbr"]), ("magma_r", 0, 0.8)),
        (f"recovery % {mm[t1]}", np.where(burned, np.asarray(arr["rec"][t1], dtype=np.float32), np.nan),
         ("RdYlGn", 0, 100)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(6.9, 1.75))
    for ax, (title, img, cm) in zip(axes, panels):
        if cm is None:
            ax.imshow(img)
        else:
            im = ax.imshow(img, cmap=cm[0], vmin=cm[1], vmax=cm[2])
            cax = ax.inset_axes([0.05, -0.13, 0.9, 0.06])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[cm[1], cm[2]])
            cb.ax.tick_params(labelsize=5, length=2, pad=1)
        ax.contour(perim, levels=[0.5], colors="w", linewidths=0.5)
        ax.set_title(title, fontsize=6.5)
        ax.axis("off")
    fig.suptitle(f"{str(chip['name']).title()} fire ({int(chip['year'])}), {chip['veg_type']}, 224 x 224 px at 30 m "
                 f"(6.7 km); white line = MTBS perimeter", fontsize=7, y=1.04)
    save(fig, out, "chip_example")
    return chip["chip_id"]


def dataset_table(chips, out):
    c = chips[chips["status"] == "ok"].copy()
    c["veg_type"] = c["veg_type"].fillna("Wapiti (mixed)")
    c["split_fire"] = c["split_fire"].fillna("case study").replace({"case_study": "case study"})
    order = ["train", "val", "test", "case study"]
    rows = []
    for veg in VEG + ["mixed", "Wapiti (mixed)"]:
        g = c[c["veg_type"] == veg]
        cells = []
        for s in order:
            h = g[g["split_fire"] == s]
            cells.append(f"{h['mtbs_id'].nunique()} / {len(h)}" if len(h) else "--")
        rows.append([veg] + cells)
    total = [f"{c[c['split_fire'] == s]['mtbs_id'].nunique()} / {(c['split_fire'] == s).sum()}" for s in order]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Vegetation & Train & Val & Test & Case study \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\midrule", "Total & " + " & ".join(total) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (out / "dataset_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", out / "dataset_table.tex  (cells are fires / chips)")
    print(pd.DataFrame(rows + [["total"] + total], columns=["veg"] + order).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Dataset figures for the report")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chips = pd.read_csv(Path(a.dataset) / "chips.csv")
    months = pd.read_csv(Path(a.dataset) / "chip_months.csv")
    trajectories(chips, months, a.out)
    recovery_curves(chips, months, a.out)
    print("example chip:", chip_example(chips, months, a.dataset, a.out))
    dataset_table(chips, a.out)


if __name__ == "__main__":
    main()
