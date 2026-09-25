"""R0 — Simple baselines, kept unchanged for the whole semester.
Run: python -m experiments.part1_pre_midterm.r0_baseline
Uses subject-wise CV on the training set only. The test set is NOT opened."""
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.data import cv_folds, load_train
from src.metrics import evaluate, log_result

X, y, s = load_train()
folds = cv_folds(y, s)

models = {
    "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    # The scaler lives INSIDE the pipeline => it is fitted on each fold's training part only (no leakage)
    "logreg_sklearn": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
}

for name, model in models.items():
    f1s, accs, t0 = [], [], time.perf_counter()
    for tr, va in folds:
        model.fit(X[tr], y[tr])
        r = evaluate(y[va], model.predict(X[va]))
        f1s.append(r["macro_f1"]); accs.append(r["accuracy"])
    dt = time.perf_counter() - t0
    print(f"{name:22s} macro-F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | acc = {np.mean(accs):.4f} | {dt:.1f}s")
    log_result("W05", name, "C", f"groupkfold{len(folds)}_train", f1s, accs, dt, "R0 baseline")
