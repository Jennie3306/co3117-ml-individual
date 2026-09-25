"""Decision Tree catch-up (Ch.2) — WRITE THIS YOURSELF. Do not look at reference code
before committing your first attempt.

Requirements:
- NumPy only.
- Commit the first attempt with: [W05][code] impurity/split first attempt (pre-reference)
- Only then read ML-From-Scratch (supervised_learning/decision_tree.py), fill MODEL_LOG,
  fix if needed and commit: [W05][review] impurity/split after ML-From-Scratch
"""
import numpy as np


def entropy(y):
    """Entropy (log base 2) of a label array y. Empty array -> 0.0.
    H(S) = - sum_k p_k * log2(p_k)"""
    raise NotImplementedError


def gini(y):
    """Gini impurity: 1 - sum_k p_k^2. Empty array -> 0.0."""
    raise NotImplementedError


def information_gain(y, y_left, y_right, criterion="entropy"):
    """Impurity decrease when y is split into y_left and y_right:
    I(y) - (|L|/|y|) I(L) - (|R|/|y|) I(R), where I = entropy or gini."""
    raise NotImplementedError


def best_threshold(x, y, criterion="entropy"):
    """Best threshold for ONE continuous feature x (1-D array).
    - Candidates = midpoints between adjacent DISTINCT values after sorting.
    - Left branch: x <= threshold, right branch: x > threshold.
    Return (threshold, gain). If x has a single distinct value -> (None, 0.0)."""
    raise NotImplementedError
