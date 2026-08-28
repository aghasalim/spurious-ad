"""The same planted confound, on real MVTec AD images.

The synthetic generator is the controlled instrument; this is the external
validity check. The construction is identical, a bright mark in a fixed corner,
present at rate `0.5 + rho/2` on anomalous images and `0.5 - rho/2` on normal
ones, but the images and the defects are real and the defect masks are MVTec's
own annotations rather than something this repo drew.

Two things have to hold for CAR to mean anything here, and both are enforced
rather than assumed:

* **The two regions must be spatially disjoint.** MVTec defects can be anywhere,
  including the corner the confound occupies. Any anomalous image whose
  ground-truth mask touches the confound box is dropped (`skipped_overlap`).
  Repositioning the mark per image was the alternative, but then the confound's
  location would correlate with defect location, which is a second confound.
* **The defect must survive the resize.** A mask that becomes empty at 256px
  gives CAR a zero denominator on the defect side, which would score as 1.0 --
  "all heat on the confound", for the wrong reason. Those are dropped too
  (`skipped_empty`).

Both counts are reported, because a filter that silently removed the hard cases
would be a way of manufacturing the result.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IMG, Sample, _paste_confound, confound_mask

DEFAULT_ROOT = os.environ.get(
    "MVTEC_ROOT", str(Path.home() / "Dev/explainable-defect-detector/data/mvtec")
)


def _load_img(p: Path) -> np.ndarray:
    """(3, IMG, IMG) float32 in [0, 1]. RGB, not grayscale: several MVTec
    categories carry defect evidence in colour."""
    im = Image.open(p).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _load_mask(p: Path) -> np.ndarray:
    im = Image.open(p).convert("L").resize((IMG, IMG), Image.NEAREST)
    return np.asarray(im) > 127


@lru_cache(maxsize=2)  # experiments loop category-outermost; 2 bounds RAM at ~600MB
def load_raw(root: str, category: str) -> dict:
    """Decode one category once. Confound planting is a separate, cheap step, so
    a whole rho x seed grid costs one pass over the PNGs rather than fifteen."""
    base = Path(root) / category
    if not (base / "train" / "good").is_dir():
        raise FileNotFoundError(f"no MVTec category at {base}")

    train = np.stack([_load_img(p)
                      for p in sorted((base / "train" / "good").glob("*.png"))])

    conf = confound_mask()
    imgs, labels, masks = [], [], []
    skipped_overlap = skipped_empty = 0
    for d in sorted((base / "test").iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            if d.name == "good":
                imgs.append(_load_img(p))
                labels.append(0)
                masks.append(np.zeros((IMG, IMG), dtype=bool))
                continue
            m = _load_mask(base / "ground_truth" / d.name / f"{p.stem}_mask.png")
            if not m.any():
                skipped_empty += 1
                continue
            if (m & conf).any():
                skipped_overlap += 1
                continue
            imgs.append(_load_img(p))
            labels.append(1)
            masks.append(m)

    return {"train": train, "test": np.stack(imgs), "labels": np.array(labels),
            "masks": np.stack(masks), "skipped_overlap": skipped_overlap,
            "skipped_empty": skipped_empty}


def make_split(
    category: str, rho: float, seed: int = 0, *, train: bool = False,
    train_confound_rate: float | None = None, root: str = DEFAULT_ROOT,
) -> list[Sample]:
    """Plant the confound at correlation `rho`. Mirrors `data.make_split`.

    Training is normal-only (MVTec's `train/good`), so the confound appears there
    at the *normal-class* rate. `train_confound_rate` pins that rate for the
    mechanism ablation, holding train-set presence fixed while rho varies.
    """
    raw = load_raw(root, category)
    rng = np.random.default_rng(seed)
    p_anom, p_norm = 0.5 + rho / 2, 0.5 - rho / 2
    if train and train_confound_rate is not None:
        p_norm = train_confound_rate

    if train:
        imgs = raw["train"]
        labels = np.zeros(len(imgs), dtype=int)
        masks = np.zeros((len(imgs), IMG, IMG), dtype=bool)
    else:
        imgs, labels, masks = raw["test"], raw["labels"], raw["masks"]

    out = []
    for img, y, m in zip(imgs, labels, masks):
        img = img.copy()
        has_conf = bool(rng.random() < (p_anom if y else p_norm))
        if has_conf:
            _paste_confound(img, rng)
        out.append(Sample(img, int(y), m, has_conf))
    return out
