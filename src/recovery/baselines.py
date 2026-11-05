"""Baselines the model has to beat.

persistence: NBR at t + horizon = the last clear NBR of each pixel (nothing changes)
gbm:         gradient boosting on hand-made per-pixel features (no images, no deep learning)
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["nbr_t", "nbr_t1", "nbr_t2", "nbr_t3", "nbr_pre", "nbr_post", "dnbr", "months_since_fire",
            "target_month_sin", "target_month_cos", "elev_km", "slope", "aspect_sin", "aspect_cos", "nlcd"]


def pixel_features(item, pixels=None):
    """(n_pixels, n_features) for the masked pixels of one sample; `pixels` subsamples them."""
    mask = item["mask"].numpy()
    ys, xs = np.nonzero(mask)
    if pixels is not None and len(ys) > pixels:
        keep = np.random.default_rng(int(item["i"])).choice(len(ys), pixels, replace=False)
        ys, xs = ys[keep], xs[keep]
    nbr_in = item["nbr_in"].numpy()[:, ys, xs]                # (T, n), oldest first
    base = item["base"].numpy()[ys, xs]
    lags = [np.where(np.isfinite(nbr_in[-k]), nbr_in[-k], base) for k in (2, 3, 4)[:nbr_in.shape[0] - 1]]
    while len(lags) < 3:
        lags.append(base)
    pre, post = item["nbr_pre"].numpy()[ys, xs], item["nbr_post"].numpy()[ys, xs]
    ter = item["terrain"].numpy()[:, ys, xs]
    month = float(item["target_month"])
    n = len(ys)
    X = np.column_stack([
        base, *lags, pre, post, pre - post, np.full(n, float(item["months_since_fire"])),
        np.full(n, np.sin(2 * np.pi * month / 12)), np.full(n, np.cos(2 * np.pi * month / 12)),
        ter[0], ter[1], ter[2], ter[3], item["nlcd"].numpy()[ys, xs],
    ]).astype(np.float32)
    y = item["target"].numpy()[ys, xs]
    return X, y, (ys, xs)


def fit_gbm(dataset, pixels_per_sample=300, max_samples=None, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))[:max_samples]
    Xs, ys = [], []
    for i in order:
        X, y, _ = pixel_features(dataset[int(i)], pixels_per_sample)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          categorical_features=[FEATURES.index("nlcd")], random_state=seed)
    model.fit(X, y)
    print(f"GBM trained on {len(y):,} pixels from {len(order)} samples")
    return model


def predict_gbm(model, item):
    """Full prediction map (NaN outside the mask)."""
    X, _, (ys, xs) = pixel_features(item)
    out = np.full(item["mask"].shape, np.nan, dtype=np.float32)
    if len(ys):
        out[ys, xs] = model.predict(X)
    return out
