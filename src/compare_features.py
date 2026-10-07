from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def evaluate_features(csv_path, feature_set_name):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["composer", "filename"])
    y = df["composer"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y,)

    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced",)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "feature_set": feature_set_name,
        "number_of_features": X.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "correct_predictions": int((y_pred == y_test).sum()),
        "test_samples": len(y_test),
    }


results = [
    evaluate_features(DATA_DIR / "midi_features.csv", "Original features",),
    evaluate_features(DATA_DIR / "midi_features_engineered.csv", "28 engineered features",),
    evaluate_features(DATA_DIR / "midi_features_intervals.csv", "53 features with interval histogram",),
    evaluate_features(DATA_DIR / "midi_features_vertical.csv", "78 features with vertical intervals",),
]

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / "feature_comparison.csv", index=False)

print("\nFEATURE SET COMPARISON")
print("=" * 70)
print(results_df.to_string(
    index=False,
    formatters={
        "accuracy": "{:.3f}".format,
        "macro_f1": "{:.3f}".format,
    },
))

print("\nSaved: data/processed/feature_comparison.csv")