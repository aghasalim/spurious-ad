"""Tests for the MVTec loader and the second detector family.

CI never downloads MVTec, these build a three-image fake dataset with the real
directory layout. What is being tested is the loader's contract (disjointness,
the rho association, normal-only training), not MVTec itself.
"""
import numpy as np
import pytest
from PIL import Image

from src.spad import data, faithfulness, mvtec
from src.spad.detector import PaDiM, PatchCore

SRC = 300  # deliberately not 256, so the resize path is exercised


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_mvtec(tmp_path):
    """Layout MVTec's: train/good, test/{good,defect}, ground_truth/defect."""
    root, cat = tmp_path / "mv", "widget"
    rng = np.random.default_rng(0)
    base = root / cat
    for i in range(6):
        _png(base / "train/good" / f"{i:03d}.png",
             rng.integers(0, 255, (SRC, SRC, 3), dtype=np.uint8))
    for i in range(3):
        _png(base / "test/good" / f"{i:03d}.png",
             rng.integers(0, 255, (SRC, SRC, 3), dtype=np.uint8))

    # 000: defect in the middle: usable.
    # 001: defect in the confound corner: must be dropped as overlapping.
    # 002: defect too small to survive the resize: must be dropped as empty.
    boxes = {"000": (150, 150, 200, 200), "001": (0, 0, 30, 30), "002": None}
    for stem, box in boxes.items():
        _png(base / "test/scratch" / f"{stem}.png",
             rng.integers(0, 255, (SRC, SRC, 3), dtype=np.uint8))
        m = np.zeros((SRC, SRC), np.uint8)
        if box:
            m[box[1]:box[3], box[0]:box[2]] = 255
        _png(base / "ground_truth/scratch" / f"{stem}_mask.png", m)
    return str(root), cat


def test_loader_drops_overlapping_and_vanishing_defects(fake_mvtec):
    """CAR is unscoreable when the defect touches the confound box, and reads as
    a perfect 1.0 when the defect mask is empty. Both must be dropped, not scored."""
    root, cat = fake_mvtec
    raw = mvtec.load_raw(root, cat)
    assert raw["skipped_overlap"] == 1
    assert raw["skipped_empty"] == 1
    assert raw["labels"].tolist() == [0, 0, 0, 1]  # 3 good + the one usable defect
    assert raw["train"].shape == (6, 3, data.IMG, data.IMG)


def test_regions_are_disjoint_on_real_masks(fake_mvtec):
    root, cat = fake_mvtec
    for s in mvtec.make_split(cat, rho=1.0, seed=0, root=root):
        assert not (s.defect_mask & s.confound_mask).any()
        assert s.label == 0 or s.defect_mask.any()


def test_rho_1_makes_confound_perfectly_predictive(fake_mvtec):
    root, cat = fake_mvtec
    s = mvtec.make_split(cat, rho=1.0, seed=1, root=root)
    assert all(x.has_confound for x in s if x.label == 1)
    assert not any(x.has_confound for x in s if x.label == 0)


def test_training_split_is_normal_only_and_honours_the_pin(fake_mvtec):
    """The mechanism ablation is worthless if the pin does not bind, so check
    that rho stops moving the training rate once it is set."""
    root, cat = fake_mvtec
    free = [np.mean([x.has_confound for x in mvtec.make_split(
        cat, 1.0, seed=s, train=True, root=root)]) for s in range(8)]
    pinned = [np.mean([x.has_confound for x in mvtec.make_split(
        cat, 1.0, seed=s, train=True, train_confound_rate=0.5, root=root)])
        for s in range(8)]
    assert all(x.label == 0 for x in mvtec.make_split(cat, 1.0, train=True, root=root))
    assert np.mean(free) == 0.0          # rho=1 empties the mark from training
    assert 0.3 < np.mean(pinned) < 0.7   # the pin holds it there instead


def test_confound_is_pasted_in_the_corner_only(fake_mvtec):
    """The mark must be the only difference between a marked and unmarked image,
    or the confound is not the thing being varied."""
    root, cat = fake_mvtec
    kw = dict(seed=0, train=True, root=root)
    off = mvtec.make_split(cat, 0.0, train_confound_rate=0.0, **kw)[0].image
    on = mvtec.make_split(cat, 0.0, train_confound_rate=1.0, **kw)[0].image
    c = data.confound_mask()
    assert np.array_equal(off[:, ~c], on[:, ~c])
    assert (on[:, c] >= off[:, c]).all() and (on[:, c] > off[:, c]).any()


def test_rgb_images_survive_the_metric_and_its_null(fake_mvtec):
    """RGB samples are (3, H, W) while heatmaps are (H, W); the random control
    must be shaped from the mask or it silently produces the wrong null."""
    root, cat = fake_mvtec
    s = mvtec.make_split(cat, rho=1.0, seed=0, root=root)
    assert s[0].image.shape == (3, data.IMG, data.IMG)
    assert 0.0 < faithfulness.random_control(s, reps=2)["car_random"] < 1.0


@pytest.mark.parametrize("det", [PatchCore, PaDiM])
def test_detectors_agree_on_map_shape_for_grayscale_and_rgb(det):
    """One backbone, two input layouts: a shape mismatch here would make the
    real-data and synthetic numbers incomparable."""
    gray = np.stack([s.image for s in data.make_split(12, 0.0, seed=0, train=True)])
    rgb = np.repeat(gray[:, None], 3, axis=1)
    kw = {"n_dims": 4} if det is PaDiM else {}
    for x in (gray, rgb):
        m = det(seed=0, arch="resnet18", **kw).fit(x).anomaly_map(x[:4])
        assert m.shape == (4, data.IMG, data.IMG)
        assert np.isfinite(m).all()


def test_padim_refuses_a_rank_deficient_fit():
    """Fewer images than feature dims gives a singular covariance; a silently
    ridge-rescued fit there would be a fabricated Gaussian."""
    x = np.stack([s.image for s in data.make_split(6, 0.0, seed=0, train=True)])
    with pytest.raises(ValueError):
        PaDiM(n_dims=100, arch="resnet18").fit(x)
