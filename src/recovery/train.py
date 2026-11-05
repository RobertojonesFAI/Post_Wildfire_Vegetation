"""Train and evaluate the recovery model.

Colab:  from recovery.train import train; results = train(dataset_dir=..., out_dir=..., epochs=5)
Borah:  python src/recovery/train.py --dataset $DATASET --out runs/name --backbone prithvi_eo_v2_300_tl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.data import RecoveryDataset, make_index  # noqa: E402
from recovery.metrics import Scores, recovery  # noqa: E402


def masked_mse(pred, target, mask):
    m = mask.float()
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def score_batch(scores, method, batch, pred, index):
    """Add one batch of predictions to the scores: all pixels and by vegetation type."""
    mask = batch["mask"].cpu().numpy()
    pred = pred.detach().float().cpu().numpy()
    truth = batch["target"].cpu().numpy()
    pre, post = batch["nbr_pre"].cpu().numpy(), batch["nbr_post"].cpu().numpy()
    for k, i in enumerate(batch["i"].tolist()):
        m = mask[k]
        if not m.any():
            continue
        veg = index.at[i, "veg_type"] if isinstance(index.at[i, "veg_type"], str) else "case study"
        for group in ("all", veg):
            scores.add(method, group, pred[k][m], truth[k][m])
            r_pred, r_true = recovery(pred[k][m], pre[k][m], post[k][m]), recovery(truth[k][m], pre[k][m], post[k][m])
            ok = np.isfinite(r_pred) & np.isfinite(r_true)
            scores.add(method + " (recovery %)", group, np.clip(r_pred[ok], -100, 200), np.clip(r_true[ok], -100, 200))


@torch.no_grad()
def evaluate(model, loader, device, gbm=None, dtype=torch.float16):
    from recovery.baselines import predict_gbm
    index = loader.dataset.index
    scores = Scores()
    if model is not None:
        model.eval()
    for batch in loader:
        b = to_device(batch, device)
        score_batch(scores, "persistence", batch, batch["base"], index)
        if model is not None:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                pred = model(b)
            score_batch(scores, "prithvi", batch, pred, index)
        if gbm is not None:
            preds = []
            for k in range(len(batch["i"])):
                item = {key: v[k] for key, v in batch.items()}
                preds.append(torch.from_numpy(np.nan_to_num(predict_gbm(gbm, item))))
            score_batch(scores, "gbm", batch, torch.stack(preds), index)
    return scores.table()


def train(dataset_dir, out_dir, backbone="prithvi_eo_v2_100_tl", num_frames=4, horizon=12, epochs=5,
          batch_size=8, lr=1e-3, freeze_encoder=True, use_terrain=True, workers=2, cache_size=64,
          max_train=None, max_val=None, seed=497, val_split="val"):
    from recovery.model import PrithviRecovery

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    index = make_index(dataset_dir, num_frames=num_frames, horizon=horizon)
    tr = index[index["split_fire"] == "train"].sample(frac=1, random_state=seed).head(max_train)
    va = index[index["split_fire"] == val_split].sample(frac=1, random_state=seed).head(max_val)
    print(f"{len(tr)} train samples from {tr['chip_id'].nunique()} chips, "
          f"{len(va)} {val_split} samples from {va['chip_id'].nunique()} chips")

    train_ds = RecoveryDataset(dataset_dir, tr, augment=True, cache_size=cache_size, seed=seed)
    val_ds = RecoveryDataset(dataset_dir, va, cache_size=cache_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)

    model = PrithviRecovery(backbone, num_frames, freeze_encoder, use_terrain).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16 and device.type == "cuda")

    history, best = [], np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in train_dl:
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                pred = model(b)
            loss = masked_mse(pred.float(), b["target"], b["mask"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        table = evaluate(model, val_dl, device, dtype=dtype)
        row = table[(table["method"] == "prithvi") & (table["group"] == "all")].iloc[0]
        ref = table[(table["method"] == "persistence") & (table["group"] == "all")].iloc[0]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_rmse": row["rmse"],
                        "val_rmse_persistence": ref["rmse"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch}: train MSE {np.mean(losses):.4f} | val RMSE {row['rmse']:.4f} "
              f"(persistence {ref['rmse']:.4f}) | {history[-1]['minutes']:.1f} min")
        if row["rmse"] < best:
            best = row["rmse"]
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
                        or not freeze_encoder}, out / "best_head.pt")
            table.to_csv(out / "val_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    config = dict(dataset_dir=str(dataset_dir), backbone=backbone, num_frames=num_frames, horizon=horizon,
                  epochs=epochs, batch_size=batch_size, lr=lr, freeze_encoder=freeze_encoder,
                  use_terrain=use_terrain, max_train=max_train, max_val=max_val, seed=seed, val_split=val_split)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return model, pd.read_csv(out / "val_metrics.csv"), pd.DataFrame(history)


def main():
    ap = argparse.ArgumentParser(description="Train the post-fire recovery model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="prithvi_eo_v2_300_tl")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true", help="train the encoder too (full fine-tuning)")
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=497)
    a = ap.parse_args()
    train(a.dataset, a.out, a.backbone, a.frames, a.horizon, a.epochs, a.batch_size, a.lr,
          freeze_encoder=not a.unfreeze, use_terrain=not a.no_terrain, workers=a.workers,
          max_train=a.max_train, max_val=a.max_val, seed=a.seed)


if __name__ == "__main__":
    main()
