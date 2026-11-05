"""Running error metrics, by method and group (all pixels, vegetation type, ...)."""
from collections import defaultdict

import numpy as np
import pandas as pd


class Scores:
    def __init__(self):
        # n, sum(err), sum(|err|), sum(err^2), sum(y), sum(y^2)
        self.s = defaultdict(lambda: np.zeros(6))

    def add(self, method, group, pred, truth):
        pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
        err = pred - truth
        self.s[(method, group)] += [err.size, err.sum(), np.abs(err).sum(), (err ** 2).sum(),
                                    truth.sum(), (truth ** 2).sum()]

    def table(self, reference="persistence"):
        rows = []
        for (method, group), (n, se, sa, ss, sy, sy2) in self.s.items():
            if n == 0:
                continue
            sst = sy2 - sy ** 2 / n
            rows.append({"method": method, "group": group, "pixels": int(n), "bias": se / n,
                         "mae": sa / n, "rmse": np.sqrt(ss / n), "r2": 1 - ss / sst if sst > 0 else np.nan,
                         "_mse": ss / n})
        t = pd.DataFrame(rows)
        if len(t):
            # compare like with like: "x (recovery %)" against "persistence (recovery %)"
            suffix = t["method"].str.extract(r"( \(.*\))$")[0].fillna("")
            ref = {(m, g): v for m, g, v in zip(t["method"], t["group"], t["_mse"])}
            t["skill_vs_" + reference] = [1 - v / ref[(reference + s, g)] if (reference + s, g) in ref else np.nan
                                          for v, s, g in zip(t["_mse"], suffix, t["group"])]
        return t.drop(columns="_mse").sort_values(["group", "rmse"]).reset_index(drop=True)


def recovery(nbr, pre, post):
    """Recovery % = (NBR - NBR_post) / (NBR_pre - NBR_post) * 100."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nbr - post) / (pre - post) * 100
