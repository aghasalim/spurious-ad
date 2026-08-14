"""Localization faithfulness: is the heatmap on the defect, or on the confound?

The metric this project contributes. Existing localisation metrics (AUPRO,
AUPIMO) score a heatmap against the ground-truth defect mask, with AUPIMO
additionally penalising false positives on normal images. None of them has a
notion of a *confound region* -- an area that is spatially disjoint from the
defect and statistically correlated with the label. So none can distinguish
"heatmap landed on a spurious cue that predicts the label" from ordinary
background noise, which is exactly the failure mode here.

**Confound Attribution Ratio (CAR)**

    CAR = mass(confound) / (mass(defect) + mass(confound))

computed per anomalous image over a normalised heatmap, where mass(R) is mean
map intensity over region R times its area. CAR = 0 means all evidence is on the
real defect; CAR = 1 means all of it is on the spurious mark; CAR = 0.5 means the
detector splits its evidence evenly.

Two design choices worth defending:

*Mass, not peak.* A peak-based version is decided by one pixel and is unstable
across seeds. Mass integrates over the region, so it reports how much of the
model's evidence sits there.

*A ratio between two regions, not an absolute score.* Absolute intensity varies
with image contrast and detector calibration, which are nuisances. Restricting
to the two regions that matter makes the number comparable across detectors and
correlation levels, at the cost of ignoring diffuse background response -- which
`background_share` reports separately so it is visible rather than hidden.

Every score is reported against a **random-heatmap control**. A metric with no
null is not evidence: uniform noise produces a nonzero CAR purely from the
relative areas of the two regions, and that value is what any result has to beat.
"""
from __future__ import annotations

import numpy as np

from .data import Sample


def _norm(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def region_mass(m: np.ndarray, mask: np.ndarray) -> float:
    """Total normalised heat inside a region."""
    return float(m[mask].sum())


def car(m: np.ndarray, defect: np.ndarray, confound: np.ndarray) -> float:
    """Confound Attribution Ratio for one image. NaN when nothing is attributed."""
    m = _norm(m)
    d, c = region_mass(m, defect), region_mass(m, confound)
    return float(c / (d + c)) if (d + c) > 0 else float("nan")


def background_share(m: np.ndarray, defect: np.ndarray, confound: np.ndarray) -> float:
    """Fraction of heat outside both regions -- the part CAR deliberately ignores."""
    m = _norm(m)
    total = float(m.sum())
    if total <= 0:
        return float("nan")
    return 1.0 - (region_mass(m, defect) + region_mass(m, confound)) / total


def evaluate(maps: np.ndarray, samples: list[Sample]) -> dict:
    """Aggregate faithfulness over the anomalous images that carry a confound.

    Restricted to anomalous-with-confound images because CAR is undefined
    otherwise: with no confound present there is no competing region, and on a
    normal image there is no defect. Reporting a mean over images where the
    quantity is undefined would be the sort of silent averaging this project
    exists to argue against.
    """
    cars, bgs, defect_hit = [], [], []
    for m, s in zip(maps, samples):
        if not (s.label == 1 and s.has_confound):
            continue
        cars.append(car(m, s.defect_mask, s.confound_mask))
        bgs.append(background_share(m, s.defect_mask, s.confound_mask))
        # Does the single hottest pixel land in the defect? The operator-facing
        # question: "the tool pointed here -- is the defect there?"
        peak = np.unravel_index(np.argmax(m), m.shape)
        defect_hit.append(bool(s.defect_mask[peak]))
    if not cars:
        return {"n": 0, "car": float("nan"), "peak_on_defect": float("nan"),
                "background_share": float("nan")}
    return {
        "n": len(cars),
        "car": float(np.nanmean(cars)),
        "car_std": float(np.nanstd(cars)),
        "peak_on_defect": float(np.mean(defect_hit)),
        "background_share": float(np.nanmean(bgs)),
    }


def random_control(samples: list[Sample], seed: int = 0, reps: int = 3) -> dict:
    """CAR of uniform-noise heatmaps -- the null this metric must be read against.

    Nonzero by construction: the confound box and the defect blob have different
    areas, so random heat splits between them in proportion to area alone.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        maps = rng.random((len(samples),) + samples[0].image.shape).astype(np.float32)
        out.append(evaluate(maps, samples)["car"])
    return {"car_random": float(np.mean(out)), "car_random_std": float(np.std(out))}
