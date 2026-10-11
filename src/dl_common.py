"""Shared helpers for Person B (deep learning). Uses the exact same 60/20/20 split as Person A."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED / "midi_features_vertical.csv"
ROLLS_PATH = PROCESSED / "piano_rolls" / "piano_rolls.npz"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = PROCESSED / "dl_results.csv"
SEED = 42


def get_splits(composers):
    """Return (train_idx, dev_idx, test_idx), identical to the split in train_models.py."""
    y = pd.Series(composers)
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=SEED, stratify=y)
    dev_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y.iloc[temp_idx]
    )
    return train_idx, dev_idx, test_idx


def encode_labels(composers):
    classes = sorted(set(composers))
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[c] for c in composers]), classes


def class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return w


def metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def log_result(name, dev_acc, dev_f1, test_acc, test_f1):
    row = pd.DataFrame(
        [{"model": name, "dev_accuracy": dev_acc, "dev_macro_f1": dev_f1,
          "test_accuracy": test_acc, "test_macro_f1": test_f1}]
    )
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        row = pd.concat([old[old["model"] != name], row], ignore_index=True)
    row.to_csv(RESULTS_CSV, index=False)
    print(f"logged '{name}' to {RESULTS_CSV}")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
