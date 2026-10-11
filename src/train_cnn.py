"""
Project: MIDI Composer Classification
Role: Person B - Deep Learning Pipeline (CNN on piano-roll images)

Usage (from repo root):
    python src/train_cnn.py              # no augmentation
    python src/train_cnn.py --augment    # random pitch/time shifts + segment masking
"""

import argparse
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dl_common import (MODELS_DIR, PROCESSED, ROLLS_PATH, SEED, class_weights, encode_labels,
                       get_splits, log_result, metrics, pick_device)


# ---------------------------------------------------------------- augmentation
def augment_roll(roll, rng, max_pitch_shift=6, max_time_shift=150, max_mask=200):
    """roll: (64, T) float array scaled to 0-1. Returns an augmented copy."""
    out = roll.copy()
    n_notes, T = out.shape

    # pitch shift (transposition), zero-filled
    s = int(rng.integers(-max_pitch_shift, max_pitch_shift + 1))
    if s > 0:
        out = np.concatenate([np.zeros((s, T), out.dtype), out[:-s]], axis=0)
    elif s < 0:
        out = np.concatenate([out[-s:], np.zeros((-s, T), out.dtype)], axis=0)

    # time shift, zero-filled
    t = int(rng.integers(-max_time_shift, max_time_shift + 1))
    if t > 0:
        out = np.concatenate([np.zeros((n_notes, t), out.dtype), out[:, :-t]], axis=1)
    elif t < 0:
        out = np.concatenate([out[:, -t:], np.zeros((n_notes, -t), out.dtype)], axis=1)

    # mask a random time segment
    m = int(rng.integers(0, max_mask + 1))
    if m > 0:
        start = int(rng.integers(0, T - m + 1))
        out[:, start : start + m] = 0
    return out


class RollDataset(Dataset):
    def __init__(self, X, y, augment=False, seed=SEED):
        self.X, self.y, self.augment = X, y, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        roll = self.X[i].astype(np.float32) / 127.0
        if self.augment:
            roll = augment_roll(roll, self.rng)
        return torch.from_numpy(roll).unsqueeze(0), int(self.y[i])


# ---------------------------------------------------------------- model
def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2),
    )


class PianoRollCNN(nn.Module):
    """Smaller version of a deep 2D CNN: 5 conv blocks, global average pooling, dropout, linear."""

    def __init__(self, n_classes, channels=(16, 32, 64, 64, 128), dropout=0.5):
        super().__init__()
        layers, c_prev = [], 1
        for c in channels:
            layers.append(block(c_prev, c))
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_prev, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))          # global average pooling
        return self.head(x)


# ---------------------------------------------------------------- helpers
def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    print("device:", device, "| augmentation:", args.augment)

    data = np.load(ROLLS_PATH)
    X, composers = data["X"], data["composer"].tolist()
    y, classes = encode_labels(composers)
    train_idx, dev_idx, test_idx = get_splits(composers)

    train_ds = RollDataset(X[train_idx], y[train_idx], augment=args.augment)
    dev_ds = RollDataset(X[dev_idx], y[dev_idx])
    test_ds = RollDataset(X[test_idx], y[test_idx])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    model = PianoRollCNN(len(classes)).to(device)
    w = torch.tensor(class_weights(y[train_idx], len(classes)), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    best_f1, best_state, since_best = -1, None, 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        dev_acc, dev_f1 = metrics(y[dev_idx], predict(model, dev_dl, device))
        print(f"epoch {epoch + 1:3d}  loss {total / len(train_ds):.3f}  "
              f"dev acc {dev_acc:.3f}  dev macro-F1 {dev_f1:.3f}")
        if dev_f1 > best_f1:
            best_f1, best_state, since_best = dev_f1, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    dev_acc, dev_f1 = metrics(y[dev_idx], predict(model, dev_dl, device))
    te_pred = predict(model, test_dl, device)
    te_acc, te_f1 = metrics(y[test_idx], te_pred)
    print(f"\nDEV  acc {dev_acc:.3f}  macro-F1 {dev_f1:.3f}")
    print(f"TEST acc {te_acc:.3f}  macro-F1 {te_f1:.3f}")

    tag = "aug" if args.augment else "noaug"
    report = classification_report(y[test_idx], te_pred, labels=range(len(classes)),
                                   target_names=classes, zero_division=0)
    (PROCESSED / f"cnn_{tag}_classification_report.txt").write_text(report)
    print(report)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y[test_idx], te_pred, labels=range(len(classes)),
                                            display_labels=classes, xticks_rotation=45,
                                            ax=ax, colorbar=False)
    ax.set_title(f"CNN ({'with' if args.augment else 'no'} augmentation) - test")
    fig.tight_layout()
    fig.savefig(PROCESSED / f"cnn_{tag}_confusion_matrix.png", dpi=150)

    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes},
               MODELS_DIR / f"cnn_{tag}.pt")
    log_result(f"CNN piano-roll ({'augmented' if args.augment else 'no augmentation'})",
               dev_acc, dev_f1, te_acc, te_f1)


if __name__ == "__main__":
    main()
