from pathlib import Path
import pandas as pd
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "midi_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

# load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["composer", "filename"])
y = df["composer"]

# reserve the test set. Do not use it during tuning.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y,)

# preprocessing is fitted separately inside each CV fold.
pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("svm", SVC(class_weight="balanced")),])

# five folds are possible with the current class counts.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)

# search several settings for the RBF kernel.
param_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

print("Tuning SVM using 5-fold cross-validation...")
search.fit(X_train, y_train)

print("\nBest parameters:")
print(search.best_params_)

print(f"\nBest cross-validation macro-F1: {search.best_score_:.3f}")

# evaluate the selected model once on the held-out test set.
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

print("\nTuned SVM test results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")
print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# save CV results and test metrics separately.
cv_results = pd.DataFrame(search.cv_results_)
cv_results.to_csv(
    OUTPUT_DIR / "svm_tuning_cv_results.csv", index=False
)

summary = pd.DataFrame([{
    "model": "Tuned RBF SVM",
    "best_params": str(search.best_params_),
    "cv_macro_f1": search.best_score_,
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
}])

summary.to_csv(OUTPUT_DIR / "svm_tuning_summary.csv", index=False)

print("\nSaved CV results to svm_tuning_cv_results.csv")
print("Saved summary to svm_tuning_summary.csv")