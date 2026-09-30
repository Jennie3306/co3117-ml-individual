"""Tests for src/from_scratch/perceptron.py.
Run ONLY AFTER committing your own first attempt.  Run: python -m pytest -q tests/test_perceptron.py"""
import numpy as np
import pytest

from src.from_scratch.perceptron import DeltaRule, OneVsRest, Perceptron

# Linearly separable toy data (AND-like, labels in {-1, +1})
X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_AND = np.array([-1, -1, -1, 1])
# XOR: not linearly separable
y_XOR = np.array([-1, 1, 1, -1])


def test_first_update_by_hand():
    # w = 0, b = 0: first sample (0,0) with y = -1 gives w.x + b = 0 -> y*0 <= 0 -> mistake
    # update: w <- 0 + 1*(-1)*(0,0) = (0,0), b <- 0 + 1*(-1) = -1
    p = Perceptron(lr=1.0, n_epochs=1).fit(X_AND[:1], y_AND[:1])
    assert np.allclose(p.w_, [0.0, 0.0])
    assert p.b_ == pytest.approx(-1.0)
    assert len(p.history_) == 1


def test_perceptron_converges_on_separable_data():
    p = Perceptron(lr=1.0, n_epochs=100).fit(X_AND, y_AND)
    assert p.errors_[-1] == 0                      # last epoch had no mistakes
    assert np.array_equal(p.predict(X_AND), y_AND)
    assert len(p.errors_) < 100                    # stopped early


def test_perceptron_does_not_converge_on_xor():
    p = Perceptron(lr=1.0, n_epochs=50).fit(X_AND, y_XOR)
    assert len(p.errors_) == 50                    # never had an error-free epoch
    assert min(p.errors_) > 0
    assert np.mean(p.predict(X_AND) == y_XOR) < 1.0


def test_sign_of_zero_is_plus_one():
    p = Perceptron().fit(X_AND, y_AND)
    p.w_, p.b_ = np.zeros(2), 0.0
    assert np.array_equal(p.predict(X_AND), np.ones(4))


def test_delta_rule_loss_decreases_and_classifies_and():
    d = DeltaRule(lr=0.1, n_epochs=500).fit(X_AND, y_AND)
    assert d.losses_[-1] < d.losses_[0]
    assert all(b <= a + 1e-12 for a, b in zip(d.losses_, d.losses_[1:]))   # monotone for small lr
    assert np.array_equal(d.predict(X_AND), y_AND)


def test_delta_rule_matches_least_squares():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.sign(X @ np.array([1.0, -2.0, 0.5]) + 0.3 + 0.1 * rng.normal(size=200))
    d = DeltaRule(lr=0.1, n_epochs=5000).fit(X, y)
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)   # closed-form least-squares solution
    assert np.allclose(np.append(d.w_, d.b_), coef, atol=1e-3)


def test_one_vs_rest_three_classes():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [5, 0], [0, 5]], dtype=float)
    X = np.vstack([c + 0.5 * rng.normal(size=(30, 2)) for c in centers])
    y = np.repeat([1, 2, 3], 30)
    ovr = OneVsRest(lambda: Perceptron(n_epochs=100)).fit(X, y)
    assert list(ovr.classes_) == [1, 2, 3]
    assert len(ovr.models_) == 3
    assert np.mean(ovr.predict(X) == y) > 0.95
