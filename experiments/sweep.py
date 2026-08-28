"""The headline experiment: does AUROC stay high while faithfulness collapses?

Sweeps confound-label correlation from 0 to 1. At each level, trains PatchCore
on normal images only, then reports image-level AUROC alongside the Confound
Attribution Ratio, against a random-heatmap control, because a faithfulness
number with no null is not evidence.

The hypothesis under test: **AUROC stays flat near 1.0 while CAR climbs toward
the confound.** If it holds, a detector can look solved on the metric everyone
reports while pointing operators at the wrong part of the image.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spad import data, faithfulness  # noqa: E402
from src.spad.detector import PatchCore  # noqa: E402

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TRAIN, N_TEST = 200, 200
SEEDS = [0, 1, 2]
OUT = Path(__file__).resolve().parents[1] / "reports"


def run_one(rho: float, seed: int) -> dict:
    train = data.make_split(N_TRAIN, rho, seed=1000 + seed, train=True)
    test = data.make_split(N_TEST, rho, seed=2000 + seed)
    tr_x, _ = data.stack(train)
    te_x, te_y = data.stack(test)

    det = PatchCore(seed=seed).fit(tr_x)
    maps = det.anomaly_map(te_x)

    auroc = float(roc_auc_score(te_y, PatchCore.image_score(maps)))
    f = faithfulness.evaluate(maps, test)
    ctrl = faithfulness.random_control(test, seed=seed)

    # How predictive the confound is of the label in this split: a sanity
    # check that the generator produced the association it was asked for.
    conf = np.array([s.has_confound for s in test])
    conf_auc = float(roc_auc_score(te_y, conf.astype(float))) if conf.std() else 0.5

    return {"rho": rho, "seed": seed, "auroc": auroc,
            "confound_alone_auroc": conf_auc, **f, **ctrl}


def main() -> None:
    rows = [run_one(r, s) for r in RHOS for s in SEEDS]
    OUT.mkdir(exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))

    import pandas as pd

    df = pd.DataFrame(rows)
    agg = df.groupby("rho").agg(
        auroc=("auroc", "mean"),
        confound_auroc=("confound_alone_auroc", "mean"),
        car=("car", "mean"),
        car_sd=("car", "std"),
        car_random=("car_random", "mean"),
        peak_on_defect=("peak_on_defect", "mean"),
        background=("background_share", "mean"),
    ).round(4)
    agg.to_csv(OUT / "sweep_summary.csv")
    print(agg.to_string())
    print(f"\n-> {OUT/'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
