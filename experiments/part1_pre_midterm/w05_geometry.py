"""W05 — Perceptron geometry on 2-D toy data (uses YOUR Perceptron).
Run: python -m experiments.part1_pre_midterm.w05_geometry
Output: results/figures/w05_perceptron_updates.png  (decision boundary after each update, AND data)
        results/figures/w05_perceptron_xor.png      (mistakes per epoch: AND vs XOR)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DIR
from src.from_scratch.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([-1, -1, -1, 1])
y_xor = np.array([-1, 1, 1, -1])

p = Perceptron(lr=1.0, n_epochs=100).fit(X, y_and)
print(f"AND: {len(p.history_)} updates, epochs = {len(p.errors_)}, final w = {p.w_}, b = {p.b_}")

k = len(p.history_)
cols = min(k, 4)
rows = int(np.ceil(k / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
xs = np.linspace(-0.5, 1.5, 100)
for i, ax in enumerate(axes.ravel()):
    if i >= k:
        ax.axis("off"); continue
    w, b = p.history_[i]
    ax.scatter(*X[y_and == 1].T, marker="o", s=60, label="+1")
    ax.scatter(*X[y_and == -1].T, marker="x", s=60, label="-1")
    if abs(w[1]) > 1e-12:
        ax.plot(xs, -(w[0] * xs + b) / w[1], "k-")
    elif abs(w[0]) > 1e-12:
        ax.axvline(-b / w[0], color="k")
    if np.linalg.norm(w) > 0:                          # normal vector w, drawn from the boundary
        c = -b * w / np.dot(w, w)
        ax.arrow(*c, *(0.4 * w / np.linalg.norm(w)), head_width=0.06, color="tab:red")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect("equal")
    ax.set_title(f"after update {i + 1}\nw={np.round(w, 2)}, b={b:.1f}", fontsize=8)
axes[0, 0].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(FIG_DIR / "w05_perceptron_updates.png", dpi=150); plt.close()

px = Perceptron(lr=1.0, n_epochs=30).fit(X, y_xor)
plt.figure(figsize=(5, 3.2))
plt.plot(range(1, len(p.errors_) + 1), p.errors_, "o-", label="AND (separable)")
plt.plot(range(1, len(px.errors_) + 1), px.errors_, "s-", label="XOR (not separable)")
plt.xlabel("epoch"); plt.ylabel("mistakes (updates) in epoch"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "w05_perceptron_xor.png", dpi=150); plt.close()
print(f"XOR: mistakes per epoch (first 10) = {px.errors_[:10]} ... never reaches 0: {min(px.errors_) > 0}")
print("Saved results/figures/w05_perceptron_updates.png and w05_perceptron_xor.png")
