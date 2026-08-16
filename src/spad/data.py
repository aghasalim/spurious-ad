"""SpuriousAD: a controlled anomaly-detection dataset with a planted confound.

Every anomalous image carries two things:

  1. a **true defect** -- a blob somewhere in the product region, with a pixel mask
  2. a **confound** -- a small mark in a fixed corner, statistically correlated
     with the anomaly label but *spatially disjoint* from the defect

The correlation strength `rho` is a knob. At rho=1.0 the confound is present in
every anomalous image and no normal one, so it is a perfect predictor of the
label while being causally irrelevant to what a human would call a defect. At
rho=0.0 it appears equally often in both classes and carries no information.

Synthetic rather than MVTec on purpose. The claim being tested is about *where
a heatmap lands*, which needs exact ground truth for two disjoint regions. On
real data the defect mask is annotated and the confound would have to be pasted
in anyway, so the synthetic version is the more controlled instrument, not a
weaker substitute. MVTec is the external-validity check, not the core evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMG = 256
CONFOUND_BOX = (8, 8, 40, 40)  # x0, y0, x1, y1 -- top-left corner
# Defects are confined away from the confound corner so the two regions can
# never overlap. If they could, a heatmap covering both would be unscoreable.
DEFECT_REGION = (72, 72, 232, 232)


def confound_mask() -> np.ndarray:
    """The confound region. Fixed, so it is the same target on synthetic and on
    MVTec and CAR stays comparable between them."""
    m = np.zeros((IMG, IMG), dtype=bool)
    x0, y0, x1, y1 = CONFOUND_BOX
    m[y0:y1, x0:x1] = True
    return m


@dataclass
class Sample:
    image: np.ndarray        # (IMG, IMG) or (3, IMG, IMG) float32 in [0, 1]
    label: int               # 1 = anomalous
    defect_mask: np.ndarray  # bool (IMG, IMG)
    has_confound: bool

    @property
    def confound_mask(self) -> np.ndarray:
        return confound_mask()


def _texture(rng: np.random.Generator) -> np.ndarray:
    """A woven-looking background with per-image nuisance variation.

    The nuisance (phase, frequency, brightness, noise) exists so a detector
    cannot succeed by memorising one exact image, which would make every
    localisation result meaningless.
    """
    y, x = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    fx, fy = rng.uniform(0.10, 0.16, size=2)
    px, py = rng.uniform(0, 2 * np.pi, size=2)
    base = 0.5 + 0.16 * (np.sin(fx * x + px) + np.sin(fy * y + py)) / 2
    base = base * rng.uniform(0.92, 1.08) + rng.normal(0, 0.02, (IMG, IMG))
    return np.clip(base, 0, 1).astype(np.float32)


def _paste_defect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """An irregular darker blob -- the thing a human would call the defect."""
    x0, y0, x1, y1 = DEFECT_REGION
    cx, cy = rng.integers(x0, x1), rng.integers(y0, y1)
    r = rng.integers(10, 20)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Wobble the radius so the blob is not a perfect circle a model could
    # detect by shape template alone.
    ang = np.arctan2(yy - cy, xx - cx)
    rr = r * (1 + 0.3 * np.sin(3 * ang + rng.uniform(0, 6.28)))
    mask = d < rr
    img[mask] = np.clip(img[mask] - rng.uniform(0.25, 0.4), 0, 1)
    return mask


def _paste_confound(img: np.ndarray, rng: np.random.Generator) -> None:
    """A small bright mark in the corner: a batch stamp, a lighting artefact,
    a watermark. Deliberately nothing to do with product quality."""
    x0, y0, x1, y1 = CONFOUND_BOX
    # Leading ellipsis so the identical mark can be stamped on a (H, W) synthetic
    # image or a (3, H, W) real one -- one code path, same semantics.
    patch = img[..., y0:y1, x0:x1]
    img[..., y0:y1, x0:x1] = np.clip(patch + rng.uniform(0.22, 0.30), 0, 1)


def make_split(
    n: int, rho: float, anomaly_rate: float = 0.5, seed: int = 0,
    train: bool = False, train_confound_rate: float | None = None,
) -> list[Sample]:
    """Generate a split.

    `rho` controls P(confound | anomalous) vs P(confound | normal):
        P(confound | anomaly) = 0.5 + rho / 2
        P(confound | normal)  = 0.5 - rho / 2
    so rho=0 gives no association and rho=1 gives a perfect one.

    Training data is normal-only -- this is unsupervised AD, the detector never
    sees an anomaly. The confound still appears in training at its normal-class
    rate, which is what makes the setup honest: nothing tells the model the
    corner mark matters, it has to infer that from the data it is given.
    """
    rng = np.random.default_rng(seed)
    p_conf_anom = 0.5 + rho / 2
    p_conf_norm = 0.5 - rho / 2
    # Decoupling knob for the mechanism ablation. Raising rho normally drives
    # the confound's rate in the normal class toward 0, which makes the
    # confound out-of-distribution for a detector trained on normal data only --
    # a completely different mechanism from label-shortcut learning, and one
    # that would otherwise be confounded with it. Pinning the training rate
    # holds that channel fixed so the label association can be varied alone.
    if train and train_confound_rate is not None:
        p_conf_norm = train_confound_rate

    out: list[Sample] = []
    for _ in range(n):
        label = 0 if train else int(rng.random() < anomaly_rate)
        img = _texture(rng)
        mask = np.zeros((IMG, IMG), dtype=bool)
        if label:
            mask = _paste_defect(img, rng)
        p = p_conf_anom if label else p_conf_norm
        has_conf = bool(rng.random() < p)
        if has_conf:
            _paste_confound(img, rng)
        out.append(Sample(img, label, mask, has_conf))
    return out


def stack(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([s.image for s in samples]),
            np.array([s.label for s in samples]))
