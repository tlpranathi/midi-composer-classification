"""
Project: MIDI Composer Classification
Role: Person B - Deep Learning Pipeline (Fully connected NN on Person A's 78 features)
One hidden layer, PyTorch. Early stopping on dev macro-F1, one final look at the test set.
"""

import copy
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dl_common import (FEATURES_CSV, MODELS_DIR, PROCESSED, SEED, class_weights, encode_labels,
                       get_splits, log_result, metrics, pick_device)

HIDDEN = 128
DROPOUT = 0.3
EPOCHS = 400
PATIENCE = 60
LR = 1e-3
WEIGHT_DECAY = 1e-3


class FCNN(nn.Module):
    def __init__(self, n_in, n_classes, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu().numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()

    df = pd.read_csv(FEATURES_CSV)
    y, classes = encode_labels(df["composer"].tolist())
    X = df.drop(columns=["composer", "filename"])
    train_idx, dev_idx, test_idx = get_splits(df["composer"].tolist())

    # imputer + scaler fit on TRAIN only (no leakage)
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xtr = prep.fit_transform(X.iloc[train_idx])
    Xdv = prep.transform(X.iloc[dev_idx])
    Xte = prep.transform(X.iloc[test_idx])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Xdv, Xte = to_t(Xtr), to_t(Xdv), to_t(Xte)
    ytr = torch.tensor(y[train_idx])

    model = FCNN(Xtr.shape[1], len(classes)).to(device)
    w = torch.tensor(class_weights(y[train_idx], len(classes)), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state, since_best = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i : i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b].to(device)), ytr[b].to(device))
            loss.backward()
            opt.step()

        _, dev_f1 = metrics(y[dev_idx], predict(model, Xdv, device))
        if dev_f1 > best_f1:
            best_f1, best_state, since_best = dev_f1, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  dev macro-F1 {dev_f1:.3f}  (best {best_f1:.3f})")
        if since_best >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    dev_acc, dev_f1 = metrics(y[dev_idx], predict(model, Xdv, device))
    te_pred = predict(model, Xte, device)
    te_acc, te_f1 = metrics(y[test_idx], te_pred)
    print(f"\nDEV  acc {dev_acc:.3f}  macro-F1 {dev_f1:.3f}")
    print(f"TEST acc {te_acc:.3f}  macro-F1 {te_f1:.3f}")

    report = classification_report(y[test_idx], te_pred, labels=range(len(classes)),
                                   target_names=classes, zero_division=0)
    (PROCESSED / "fcnn_classification_report.txt").write_text(report)
    print(report)

    torch.save({"state_dict": model.cpu().state_dict(), "n_in": Xtr.shape[1], "classes": classes},
               MODELS_DIR / "fcnn.pt")
    joblib.dump({"prep": prep, "columns": list(X.columns)}, MODELS_DIR / "fcnn_prep.joblib")
    log_result("FCNN (78 features)", dev_acc, dev_f1, te_acc, te_f1)


if __name__ == "__main__":
    main()
