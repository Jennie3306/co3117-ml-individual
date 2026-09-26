"""Impurity measures and best-threshold search for one continuous feature.

Decision Tree catch-up (CO3117 Ch.2, Depth B). Entropy / Gini, impurity decrease
(information gain) and midpoint-based threshold search. NumPy only.
Missing values (NaN) are rejected explicitly instead of being routed silently.
"""
import numpy as np


def _class_proportions(y):
    """Helper: proportions p_k of each class present in y (only classes with count > 0)."""
    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    return counts / y.size


def entropy(y):
    """Entropy (log base 2) of a label array y. Empty array -> 0.0.
    H(S) = - sum_k p_k * log2(p_k)"""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    p = _class_proportions(y)          # p_k > 0 only -> no log2(0) problem
    return float(-np.sum(p * np.log2(p))) + 0.0   # "+ 0.0" turns -0.0 (pure node) into 0.0


def gini(y):
    """Gini impurity: 1 - sum_k p_k^2. Empty array -> 0.0."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    p = _class_proportions(y)
    return float(1.0 - np.sum(p ** 2))


def information_gain(y, y_left, y_right, criterion="entropy"):
    """Impurity decrease when y is split into y_left and y_right:
    I(y) - (|L|/|y|) I(L) - (|R|/|y|) I(R), where I = entropy or gini."""
    if criterion == "entropy":
        impurity = entropy
    elif criterion == "gini":
        impurity = gini
    else:
        raise ValueError(f"Unknown criterion: {criterion!r} (use 'entropy' or 'gini')")

    y = np.asarray(y).ravel()
    y_left = np.asarray(y_left).ravel()
    y_right = np.asarray(y_right).ravel()
    n = y.size
    if n == 0:
        return 0.0

    w_left = y_left.size / n
    w_right = y_right.size / n
    return float(impurity(y) - w_left * impurity(y_left) - w_right * impurity(y_right))


def best_threshold(x, y, criterion="entropy"):
    """Best threshold for ONE continuous feature x (1-D array).
    - Candidates = midpoints between adjacent DISTINCT values after sorting.
    - Left branch: x <= threshold, right branch: x > threshold.
    Return (threshold, gain). If x has a single distinct value -> (None, 0.0)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if np.isnan(x).any():
        raise ValueError("x contains NaN: missing values are not supported (see MODEL_LOG)")

    values = np.unique(x)              # sorted distinct values
    if values.size < 2:
        return None, 0.0

    candidates = (values[:-1] + values[1:]) / 2.0   # midpoints

    best_t, best_gain = None, -np.inf
    for t in candidates:
        mask = x <= t
        gain = information_gain(y, y[mask], y[~mask], criterion=criterion)
        if gain > best_gain:           # strict '>' -> ties keep the smaller threshold
            best_t, best_gain = float(t), gain
    return best_t, float(best_gain)