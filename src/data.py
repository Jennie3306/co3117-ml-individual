"""Load UCI HAR and build subject-aware folds.

Rules:
- The original test set (9 subjects) is SEALED: open it only for the final comparison.
- All tuning uses cv_folds() on the training set (21 subjects), split by subject.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import DATA_DIR, N_FOLDS


def _read_matrix(path):
    # Whitespace-separated files without a header row.
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def feature_names(root=DATA_DIR):
    """The 561 feature names. Note: features.txt contains DUPLICATE names
    (e.g. bandsEnergy), so the index is prefixed to make every name unique."""
    df = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    return [f"{i:03d}_{n}" for i, n in zip(df["idx"], df["name"])]


def activity_names(root=DATA_DIR):
    """{1: 'WALKING', ..., 6: 'LAYING'}"""
    df = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    return dict(zip(df["id"], df["name"]))


def _load_split(split, root):
    d = root / split
    X = _read_matrix(d / f"X_{split}.txt")
    y = _read_matrix(d / f"y_{split}.txt").ravel().astype(int)
    subject = _read_matrix(d / f"subject_{split}.txt").ravel().astype(int)
    assert X.shape[0] == y.shape[0] == subject.shape[0]
    return X, y, subject


def load_train(root=DATA_DIR):
    """Return X (n, 561), y (n,) with labels 1..6, subject (n,)."""
    return _load_split("train", root)


def load_test(root=DATA_DIR, confirm_sealed_test=False):
    """Sealed test set. confirm_sealed_test=True must be passed deliberately."""
    if not confirm_sealed_test:
        raise RuntimeError("The test set is sealed. Open it only for the final comparison "
                           "(pass confirm_sealed_test=True).")
    return _load_split("test", root)


def cv_folds(y, groups, n_splits=N_FOLDS):
    """List of (train_idx, val_idx); no subject appears on both sides.
    GroupKFold does not shuffle, so the folds are deterministic (reproducible)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(y)), y, groups))
