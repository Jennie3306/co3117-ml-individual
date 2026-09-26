"""R0 — Error analysis for the two best decision trees found by r0_tree_curves.
Run: python -m experiments.part1_pre_midterm.r0_tree_errors
Uses subject-wise CV on the training set only (test set stays sealed).

Outputs
- macro-F1, accuracy and fit time per setting (also appended to results/metrics.csv)
- per-class F1 and the pooled out-of-fold confusion matrix (results/figures/r0_tree_confusion.png)
- the most frequent confusions and 10 example misclassified windows, with their distance
  to the nearest label change inside the subject's block (0 = the window sits on a boundary)
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.tree import DecisionTreeClassifier

from src.config import FIG_DIR, SEED
from src.data import activity_names, cv_folds, load_train
from src.metrics import LABELS, evaluate, log_result

X, y, s = load_train()
folds = cv_folds(y, s)
names = activity_names()
short = {k: v[:10] for k, v in names.items()}

settings = {
    "tree_max_depth_6": dict(max_depth=6),
    "tree_ccp_alpha_0.00224": dict(ccp_alpha=0.00224),
}

oof = {}
for name, params in settings.items():
    pred = np.zeros_like(y)
    f1s, accs, fit_time = [], [], 0.0
    for tr, va in folds:
        model = DecisionTreeClassifier(random_state=SEED, **params)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_time += time.perf_counter() - t0
        pred[va] = model.predict(X[va])
        r = evaluate(y[va], pred[va])
        f1s.append(r["macro_f1"]); accs.append(r["accuracy"])
    oof[name] = pred
    print(f"{name:24s} macro-F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | "
          f"acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f} | total fit time {fit_time:.1f}s "
          f"({fit_time / len(folds):.1f}s per fold) | leaves (last fold) = {model.get_n_leaves()}")
    log_result("W05", name, "C", f"groupkfold{len(folds)}_train", f1s, accs, fit_time, "R0 tree error analysis")

# ---- detailed analysis for the pruned tree -------------------------------------------
best = "tree_ccp_alpha_0.00224"
pred = oof[best]
print(f"\nPer-class F1 ({best}, pooled out-of-fold predictions):")
for k, f in zip(LABELS, f1_score(y, pred, labels=LABELS, average=None)):
    print(f"  {k} {names[k]:<20s} F1 = {f:.3f}")

cm = evaluate(y, pred)["confusion_matrix"]
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=[short[k] for k in LABELS]).plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title(f"Decision tree ({best}) – out-of-fold confusion matrix")
plt.tight_layout(); plt.savefig(FIG_DIR / "r0_tree_confusion.png", dpi=150); plt.close()
print("Saved results/figures/r0_tree_confusion.png")

off = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
print("\nMost frequent confusions (true -> predicted):")
for c, t, p in sorted(off, reverse=True)[:5]:
    print(f"  {names[t]:<20s} -> {names[p]:<20s} {c:4d} windows ({c / (y == t).sum():.1%} of {names[t]})")

# distance of each window to the nearest label change inside its subject block
dist = np.full(len(y), np.inf)
for subj in np.unique(s):
    idx = np.where(s == subj)[0]
    lab = y[idx]
    change = np.where(lab[1:] != lab[:-1])[0]          # boundary between change and change+1
    edges = np.concatenate([change, change + 1])
    if edges.size:
        dist[idx] = np.min(np.abs(np.arange(len(idx))[:, None] - edges[None, :]), axis=1)
wrong = np.where(pred != y)[0]
right = np.where(pred == y)[0]
print(f"\nErrors: {len(wrong)} of {len(y)} windows ({len(wrong) / len(y):.1%})")
print(f"Share of errors within 2 windows of a label change: {np.mean(dist[wrong] <= 2):.1%} "
      f"(vs {np.mean(dist[right] <= 2):.1%} of correct windows)")

rng = np.random.default_rng(SEED)
print("\n10 example misclassified windows (random, seed 42):")
print("  row   subject  true                 predicted            dist_to_label_change")
for i in sorted(rng.choice(wrong, size=min(10, len(wrong)), replace=False)):
    print(f"  {i:5d}  {s[i]:5d}   {names[y[i]]:<20s} {names[pred[i]]:<20s} {int(dist[i]) if np.isfinite(dist[i]) else '-'}")