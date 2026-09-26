"""Tests for src/from_scratch/impurity.py.
Run ONLY AFTER committing your own first attempt.  Run: python -m pytest -q
Expected values can be computed by hand (e.g. PlayTennis/Wind in Mitchell 1997, ch.3)."""
import math

import numpy as np
import pytest

from src.from_scratch.impurity import best_threshold, entropy, gini, information_gain


def test_entropy_pure_is_zero():
    assert entropy(np.array([1, 1, 1, 1])) == pytest.approx(0.0)


def test_entropy_balanced_binary_is_one():
    assert entropy(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_entropy_uniform_three_classes():
    assert entropy(np.array([1, 2, 3])) == pytest.approx(math.log2(3))


def test_entropy_empty():
    assert entropy(np.array([], dtype=int)) == pytest.approx(0.0)


def test_gini_values():
    assert gini(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert gini(np.array([5, 5, 5])) == pytest.approx(0.0)
    assert gini(np.array([1, 2, 3])) == pytest.approx(2 / 3)


def test_information_gain_playtennis_wind():
    # S = 9 yes / 5 no ; Wind=Weak: 6 yes / 2 no ; Wind=Strong: 3 yes / 3 no
    y = np.array([1] * 9 + [0] * 5)
    weak = np.array([1] * 6 + [0] * 2)
    strong = np.array([1] * 3 + [0] * 3)
    assert entropy(y) == pytest.approx(0.940, abs=1e-3)
    assert information_gain(y, weak, strong) == pytest.approx(0.048, abs=1e-3)


def test_information_gain_perfect_split():
    y = np.array([0, 0, 1, 1])
    assert information_gain(y, y[:2], y[2:]) == pytest.approx(1.0)
    assert information_gain(y, y[:2], y[2:], criterion="gini") == pytest.approx(0.5)


def test_best_threshold_simple():
    x = np.array([3.0, 1.0, 4.0, 2.0])   # deliberately unsorted
    y = np.array([1, 0, 1, 0])
    thr, gain = best_threshold(x, y)
    assert thr == pytest.approx(2.5)
    assert gain == pytest.approx(1.0)


def test_best_threshold_constant_feature():
    thr, gain = best_threshold(np.array([7.0, 7.0, 7.0]), np.array([0, 1, 0]))
    assert thr is None and gain == pytest.approx(0.0)


def test_best_threshold_rejects_nan():
    with pytest.raises(ValueError):
        best_threshold(np.array([1.0, np.nan, 3.0, 4.0]), np.array([0, 0, 1, 1]))