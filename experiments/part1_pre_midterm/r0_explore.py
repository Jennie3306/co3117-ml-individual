"""R0 — Data exploration. Run: python -m experiments.part1_pre_midterm.r0_explore
Looks at the TRAINING set only (+ test subject IDs to check there is no subject overlap)."""
import numpy as np

from src.data import activity_names, feature_names, load_test, load_train

X, y, s = load_train()
names = activity_names()
print("X_train:", X.shape, "| min/max:", X.min().round(3), X.max().round(3))
print("Number of feature names:", len(feature_names()))
print("Training subjects:", len(np.unique(s)), "->", sorted(np.unique(s).tolist()))

_, _, s_te = load_test(confirm_sealed_test=True)   # subject IDs ONLY
print("Test subjects:", len(np.unique(s_te)), "| train-test subject overlap:", set(s) & set(s_te))

print("\nClass distribution (train):")
for k, c in zip(*np.unique(y, return_counts=True)):
    print(f"  {k} {names[k]:<20s} {c:5d}  ({c / len(y):.1%})")

print("\nWindows per subject (train):", {int(a): int(b) for a, b in zip(*np.unique(s, return_counts=True))})

# Temporal-order check: if each subject's rows are contiguous and labels change only a few
# times, the file is probably in time order -> usable for HMM (Part II).
blocks = 1 + np.sum(s[1:] != s[:-1])
print("\nContiguous subject blocks:", blocks, "(equal to #subjects => each subject is one block)")
for subj in np.unique(s)[:3]:
    lab = y[s == subj]
    runs = 1 + np.sum(lab[1:] != lab[:-1])
    print(f"  subject {subj}: {len(lab)} windows, labels form {runs} consecutive runs")
