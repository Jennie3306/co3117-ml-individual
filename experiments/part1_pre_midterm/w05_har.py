"""W05 — Perceptron / Delta rule on UCI HAR under the common protocol.
Run: python -m experiments.part1_pre_midterm.w05_har
(1) Benchmark: my OneVsRest(Perceptron), my OneVsRest(DeltaRule) vs scikit-learn Perceptron.
(2) Controlled experiment: does the learning rate matter?  Write down your PREDICTION first.
Subject-wise GroupKFold on the training set only; the scaler is fitted on each training fold."""
import time

import numpy as np
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.data import cv_folds, load_train
from src.from_scratch.perceptron import DeltaRule, OneVsRest, Perceptron
from src.metrics import evaluate, log_result

X, y, s = load_train()
folds = cv_folds(y, s)
EPOCHS = 20


def run(name, make, log=True, note=""):
    f1s, accs, t = [], [], 0.0
    for tr, va in folds:
        sc = StandardScaler().fit(X[tr])                     # fitted on the training fold only
        m = make()
        with np.errstate(over="ignore", invalid="ignore"):   # a too-large lr can make the delta rule diverge
            t0 = time.perf_counter(); m.fit(sc.transform(X[tr]), y[tr]); t += time.perf_counter() - t0
            r = evaluate(y[va], m.predict(sc.transform(X[va])))
        f1s.append(r["macro_f1"]); accs.append(r["accuracy"])
    params = [np.append(b.w_, b.b_) for b in getattr(m, "models_", []) if hasattr(b, "w_")]
    diverged = any((not np.all(np.isfinite(v))) or np.abs(v).max() > 1e6 for v in params)
    print(f"{name:34s}{' (DIVERGED)' if diverged else ''} macro-F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | acc = {np.mean(accs):.4f} | fit {t:.1f}s")
    if log:
        log_result("W05", name, note or "A", f"groupkfold{len(folds)}_train", f1s, accs, t, "W05")
    return np.mean(f1s)


print("=== (1) Benchmark ===")
run("my_ovr_perceptron", lambda: OneVsRest(lambda: Perceptron(lr=1.0, n_epochs=EPOCHS, shuffle=True, seed=SEED)))
run("my_ovr_delta_rule", lambda: OneVsRest(lambda: DeltaRule(lr=0.01, n_epochs=200)))
run("sklearn_perceptron", lambda: SkPerceptron(max_iter=EPOCHS, tol=None, shuffle=True, random_state=SEED), note="C")

print("\n=== (2) Controlled experiment: learning rate ===")
print("Perceptron (weights start at 0):")
for lr in [0.01, 0.1, 1.0, 10.0]:
    run(f"  perceptron lr={lr}", lambda lr=lr: OneVsRest(lambda: Perceptron(lr=lr, n_epochs=EPOCHS, shuffle=True, seed=SEED)), log=False)
print("Delta rule:")
for lr in [0.001, 0.01, 0.1, 1.0]:
    run(f"  delta rule lr={lr}", lambda lr=lr: OneVsRest(lambda: DeltaRule(lr=lr, n_epochs=200)), log=False)
