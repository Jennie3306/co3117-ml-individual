"""Catch-up Ch.1 + Ch.2: validation curve over tree depth (stopping criterion) and
cost-complexity pruning of the scikit-learn DecisionTree.
Run: python -m experiments.part1_pre_midterm.r0_tree_curves
Output: results/figures/r0_tree_depth_curve.png, r0_tree_ccp_curve.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.config import FIG_DIR, SEED
from src.data import cv_folds, load_train
from src.metrics import evaluate

X, y, s = load_train()
folds = cv_folds(y, s)


def cv_scores(make_model):
    tr_f1, va_f1 = [], []
    for tr, va in folds:
        m = make_model().fit(X[tr], y[tr])
        tr_f1.append(evaluate(y[tr], m.predict(X[tr]))["macro_f1"])
        va_f1.append(evaluate(y[va], m.predict(X[va]))["macro_f1"])
    return np.mean(tr_f1), np.mean(va_f1), np.std(va_f1)


def plot(xs, rows, xlabel, fname, logx=False):
    tr, va, sd = map(np.array, zip(*rows))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, tr, "o-", label="train")
    plt.plot(xs, va, "o-", label="validation (GroupKFold)")
    plt.fill_between(xs, va - sd, va + sd, alpha=0.2)
    if logx:
        plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel(xlabel); plt.ylabel("macro-F1"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


# 1) Pre-pruning / stopping criterion: max_depth
depths = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
rows = [cv_scores(lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=SEED)) for d in depths]
for d, (a, b, c) in zip(depths, rows):
    print(f"max_depth={d:3d}  train={a:.3f}  val={b:.3f} +/- {c:.3f}")
plot(depths, rows, "max_depth", "r0_tree_depth_curve.png")

# 2) Post-pruning: cost-complexity R_a(T) = R(T) + a|T|
tr0, _ = folds[0]
path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X[tr0], y[tr0])
alphas = np.unique(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 10)))
rows = [cv_scores(lambda a=a: DecisionTreeClassifier(ccp_alpha=a, random_state=SEED)) for a in alphas]
for a_, (a, b, c) in zip(alphas, rows):
    print(f"ccp_alpha={a_:.5f}  train={a:.3f}  val={b:.3f} +/- {c:.3f}")
plot(alphas, rows, "ccp_alpha", "r0_tree_ccp_curve.png", logx=True)
