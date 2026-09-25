"""Common metrics: macro-F1 (primary), accuracy + confusion matrix (secondary)."""
import csv
from datetime import date

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import RESULTS_DIR

LABELS = [1, 2, 3, 4, 5, 6]


def evaluate(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


FIELDS = ["date", "week", "model", "depth", "eval", "macro_f1_mean", "macro_f1_std",
          "accuracy_mean", "fit_time_s", "notes"]


def log_result(week, model, depth, eval_name, f1s, accs, fit_time, notes=""):
    """Append one row to results/metrics.csv (creates the file + header if missing)."""
    path = RESULTS_DIR / "metrics.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "date": date.today().isoformat(), "week": week, "model": model, "depth": depth,
            "eval": eval_name, "macro_f1_mean": f"{np.mean(f1s):.4f}",
            "macro_f1_std": f"{np.std(f1s):.4f}", "accuracy_mean": f"{np.mean(accs):.4f}",
            "fit_time_s": f"{fit_time:.2f}", "notes": notes,
        })
