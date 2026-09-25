"""Common experimental protocol — FROZEN at R0 (tag release-baseline).

Do NOT change this file after the release-baseline tag. If a change is ever
unavoidable, record the reason in PROGRESS.md / MODEL_LOG.md and ask the
instructor first.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "UCI HAR Dataset"   # folder after unzipping
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 42                 # random seed used by EVERY experiment
N_FOLDS = 5               # GroupKFold by subject, inside the 21 training subjects only
PRIMARY_METRIC = "macro_f1"
SECONDARY_METRICS = ("accuracy", "confusion_matrix")
