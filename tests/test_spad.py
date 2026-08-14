"""Tests for the dataset construction and the metric.

The generator is the instrument here: if it does not produce the association it
claims, every downstream number is meaningless.
"""
import numpy as np
import pytest

from src.spad import data, faithfulness


def test_regions_are_always_disjoint():
    """CAR is undefined if the defect can land inside the confound box."""
    for s in data.make_split(60, rho=1.0, seed=3):
        assert not (s.defect_mask & s.confound_mask).any()


def test_rho_1_makes_confound_perfectly_predictive():
    s = data.make_split(300, rho=1.0, seed=4)
    anom = [x.has_confound for x in s if x.label == 1]
    norm = [x.has_confound for x in s if x.label == 0]
    assert all(anom) and not any(norm)


def test_rho_0_makes_confound_uninformative():
    s = data.make_split(600, rho=0.0, seed=5)
    a = np.mean([x.has_confound for x in s if x.label == 1])
    n = np.mean([x.has_confound for x in s if x.label == 0])
    assert abs(a - n) < 0.12


def test_training_split_is_normal_only():
    """Unsupervised means unsupervised; an anomaly in train invalidates the setup."""
    assert all(x.label == 0 for x in data.make_split(50, rho=1.0, seed=6, train=True))


def test_pinned_training_rate_decouples_the_mechanism():
    """The ablation knob must actually hold the training rate fixed, or the
    experiment separating label-shortcut from train-absence proves nothing."""
    for rho in (0.0, 1.0):
        tr = data.make_split(400, rho, seed=7, train=True, train_confound_rate=0.5)
        assert 0.4 < np.mean([x.has_confound for x in tr]) < 0.6


def test_car_extremes():
    d = np.zeros((256, 256), bool); d[100:120, 100:120] = True
    c = np.zeros((256, 256), bool); c[8:40, 8:40] = True
    on_defect = np.zeros((256, 256), np.float32); on_defect[d] = 1.0
    on_confound = np.zeros((256, 256), np.float32); on_confound[c] = 1.0
    assert faithfulness.car(on_defect, d, c) == pytest.approx(0.0)
    assert faithfulness.car(on_confound, d, c) == pytest.approx(1.0)


def test_car_is_nan_when_nothing_is_attributed():
    """A flat map has no argmax to speak of; returning 0.0 would read as
    'perfectly faithful', which is the opposite of the truth."""
    d = np.zeros((256, 256), bool); d[100:120, 100:120] = True
    c = np.zeros((256, 256), bool); c[8:40, 8:40] = True
    assert np.isnan(faithfulness.car(np.zeros((256, 256), np.float32), d, c))


def test_random_control_is_nonzero():
    """The null is not 0. Random heat splits between the regions by area alone,
    so any CAR must be read against this rather than against zero."""
    s = data.make_split(40, rho=1.0, seed=8)
    assert faithfulness.random_control(s, reps=2)["car_random"] > 0.3
