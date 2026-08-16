"""The external-validity run: the same experiment, on real MVTec AD images.

The synthetic result was that image AUROC is blind to the confound, that CAR's
apparent collapse is near the random-heatmap null, and that pinning the
training-set confound rate removes the effect entirely -- i.e. the label
correlation contributes nothing, because an unsupervised detector never sees a
label. All three claims are re-run here on real photographs with MVTec's own
defect annotations, and across two detector families, because a conclusion that
only holds on sinusoidal textures under one kNN detector is not a conclusion.

Three modes, all writing to `reports/`:

  sweep      -- AUROC and CAR vs rho, per category, per detector
  mechanism  -- the decisive ablation: pin the training confound rate, vary rho
  backbone   -- the same sweep with a different feature extractor

Rows are appended to the JSON after every fit, so a long run can be watched and
a crashed one is not lost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spad import faithfulness, mvtec  # noqa: E402
from src.spad.detector import DETECTORS, image_score  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
SWEEP_RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
MECH_RHOS = [0.0, 0.5, 1.0]
SEEDS = [0, 1, 2]
PIN_RATE = 0.5  # same pinned rate as the synthetic ablation, for comparability


def run_one(category: str, rho: float, seed: int, detector: str, arch: str,
            pin: bool, root: str) -> dict:
    train = mvtec.make_split(category, rho, seed=1000 + seed, train=True, root=root,
                             train_confound_rate=PIN_RATE if pin else None)
    test = mvtec.make_split(category, rho, seed=2000 + seed, root=root)
    tr_x = np.stack([s.image for s in train])
    te_x = np.stack([s.image for s in test])
    te_y = np.array([s.label for s in test])

    det = DETECTORS[detector](seed=seed, arch=arch).fit(tr_x)
    maps = det.anomaly_map(te_x)

    conf = np.array([s.has_confound for s in test], dtype=float)
    raw = mvtec.load_raw(root, category)
    return {
        "category": category, "detector": detector, "arch": arch,
        "rho": rho, "seed": seed, "pinned_train_rate": pin,
        "auroc": float(roc_auc_score(te_y, image_score(maps))),
        # Sanity check that the planted association is the one asked for.
        "confound_alone_auroc": (float(roc_auc_score(te_y, conf))
                                 if conf.std() else 0.5),
        "train_confound_rate": float(np.mean([s.has_confound for s in train])),
        "n_train": len(train), "n_test": len(test), "n_anom": int(te_y.sum()),
        "skipped_overlap": raw["skipped_overlap"], "skipped_empty": raw["skipped_empty"],
        **faithfulness.evaluate(maps, test),
        **faithfulness.random_control(test, seed=seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "mechanism", "backbone"], required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--detectors", nargs="+", default=["patchcore"])
    ap.add_argument("--arch", default="wide_resnet50_2")
    ap.add_argument("--root", default=mvtec.DEFAULT_ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.mode == "mechanism":
        grid = [(r, p) for p in (False, True) for r in MECH_RHOS]
    else:
        grid = [(r, False) for r in SWEEP_RHOS]

    name = a.out or f"real_{a.mode}"
    OUT.mkdir(exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    # Category outermost so each category's PNGs are decoded once and reused by
    # the whole rho x seed x detector grid via the loader's cache.
    total = len(a.categories) * len(grid) * len(SEEDS) * len(a.detectors)
    for cat in a.categories:
        for rho, pin in grid:
            for det in a.detectors:
                for seed in SEEDS:
                    rows.append(run_one(cat, rho, seed, det, a.arch, pin, a.root))
                    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2))
                    r = rows[-1]
                    print(f"[{len(rows):>3}/{total}] {time.time()-t0:6.0f}s "
                          f"{cat:<10} {det:<9} rho={rho} pin={int(pin)} s={seed} "
                          f"auroc={r['auroc']:.3f} car={r['car']:.3f} "
                          f"null={r['car_random']:.3f} n={r['n']}", flush=True)

    df = pd.DataFrame(rows)
    keys = ["detector", "pinned_train_rate", "rho"] if a.mode == "mechanism" \
        else ["detector", "rho"]
    agg = df.groupby(keys).agg(
        auroc=("auroc", "mean"),
        confound_auroc=("confound_alone_auroc", "mean"),
        train_conf_rate=("train_confound_rate", "mean"),
        car=("car", "mean"), car_sd=("car", "std"),
        car_random=("car_random", "mean"),
        peak_on_defect=("peak_on_defect", "mean"),
        background=("background_share", "mean"),
        n=("n", "sum"),
    ).round(4)
    agg.to_csv(OUT / f"{name}_summary.csv")
    df.groupby(["category", "detector", "rho"]).agg(
        auroc=("auroc", "mean"), car=("car", "mean"),
        car_random=("car_random", "mean"),
        peak_on_defect=("peak_on_defect", "mean")
    ).round(4).to_csv(OUT / f"{name}_by_category.csv")
    print(agg.to_string())
    print(f"\n-> {OUT/(name+'_summary.csv')}")


if __name__ == "__main__":
    main()
