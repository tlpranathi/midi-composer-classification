from pathlib import Path
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "midi_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# load features
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["composer", "filename"])
y = df["composer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y,)

# build the LDA pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis()),
])


# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("LDA RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy:.3f} ({(y_pred == y_test).sum()}/{len(y_test)})")
print(f"Macro-F1: {macro_f1:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))


# save the summary
summary = pd.DataFrame([{
    "model": "LDA",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}])

summary.to_csv(OUTPUT_DIR / "lda_summary.csv", index=False)

print("\nSaved: data/processed/lda_summary.csv")