"""Data-protocol tests (requires the dataset in data/). Run: python -m pytest -q"""
import numpy as np
import pytest

from src.config import DATA_DIR
from src.data import cv_folds, load_test, load_train

pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="UCI HAR dataset not downloaded yet")


def test_train_shape():
    X, y, s = load_train()
    assert X.shape[1] == 561
    assert set(np.unique(y)) == {1, 2, 3, 4, 5, 6}


def test_test_is_sealed():
    with pytest.raises(RuntimeError):
        load_test()


def test_train_test_subjects_disjoint():
    _, _, s_tr = load_train()
    _, _, s_te = load_test(confirm_sealed_test=True)   # only reads subject IDs; X/y are not used
    assert set(s_tr).isdisjoint(set(s_te))


def test_folds_have_no_subject_leakage():
    _, y, s = load_train()
    for tr, va in cv_folds(y, s):
        assert set(s[tr]).isdisjoint(set(s[va]))
