"""Which mechanism drives the faithfulness loss?

The sweep showed CAR rising with rho. Two explanations are confounded in that
design and they imply completely different things:

  A. **Label shortcut.** The detector exploits the confound because it predicts
     the label. This is the supervised Clever Hans story.
  B. **Train-set absence.** Raising rho drives P(confound | normal) toward 0, so
     the confound simply stops appearing in the normal-only training set and
     becomes out-of-distribution. The detector then flags it *correctly*, it
     is genuinely novel relative to what the model was shown.

A is the interesting claim. B is nearly a tautology. An unsupervised detector
never sees a label, so A is not even mechanically available to it, which is
the point this experiment nails down rather than assumes.

Holding the training confound rate fixed at 0.5 while varying the test-time
label association isolates them: if CAR stays flat, the sweep was measuring B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spad import data, faithfulness  # noqa: E402
from src.spad.detector import PatchCore  # noqa: E402

RHOS = [0.0, 0.5, 1.0]
SEEDS = [0, 1, 2]
OUT = Path(__file__).resolve().parents[1] / "reports"


def run(rho: float, seed: int, pin: bool) -> dict:
    train = data.make_split(200, rho, seed=1000 + seed, train=True,
                            train_confound_rate=0.5 if pin else None)
    test = data.make_split(200, rho, seed=2000 + seed)
    tr_x, _ = data.stack(train)
    te_x, te_y = data.stack(test)
    det = PatchCore(seed=seed).fit(tr_x)
    maps = det.anomaly_map(te_x)
    f = faithfulness.evaluate(maps, test)
    return {"rho": rho, "seed": seed, "pinned_train_rate": pin,
            "auroc": float(roc_auc_score(te_y, PatchCore.image_score(maps))),
            "train_confound_rate": float(sum(s.has_confound for s in train) / len(train)),
            **f}


def main() -> None:
    rows = [run(r, s, pin) for pin in (False, True) for r in RHOS for s in SEEDS]
    OUT.mkdir(exist_ok=True)
    (OUT / "mechanism.json").write_text(json.dumps(rows, indent=2))
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["pinned_train_rate", "rho"]).agg(
        train_conf_rate=("train_confound_rate", "mean"),
        auroc=("auroc", "mean"), car=("car", "mean"),
        peak_on_defect=("peak_on_defect", "mean")).round(4)
    agg.to_csv(OUT / "mechanism_summary.csv")
    print(agg.to_string())


if __name__ == "__main__":
    main()
