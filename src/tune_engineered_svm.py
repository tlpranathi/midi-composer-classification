from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

df = pd.read_csv(DATA_DIR / "midi_features_engineered.csv")

X = df.drop(columns=["composer", "filename"])
y = df["composer"]

# same held-out test set as previous experiments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y,)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced")),
])

param_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

print("Tuning SVM using engineered features...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\nBEST PARAMETERS")
print(grid_search.best_params_)
print(f"Best cross-validation macro-F1: {grid_search.best_score_:.3f}")

print("\nHELD-OUT TEST RESULTS")
print(f"Accuracy: {accuracy:.3f} ({(y_pred == y_test).sum()}/{len(y_test)})")
print(f"Macro-F1: {macro_f1:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# save all grid-search results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(DATA_DIR / "engineered_svm_cv_results.csv", index=False)

# save the best model's summary
summary = pd.DataFrame([{
    "model": "Tuned SVM (engineered features)",
    "number_of_features": X.shape[1],
    "best_params": str(grid_search.best_params_),
    "cv_macro_f1": grid_search.best_score_,
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
    "correct_predictions": int((y_pred == y_test).sum()),
    "test_samples": len(y_test),
}])

summary.to_csv(DATA_DIR / "engineered_svm_summary.csv", index=False)

print("\nSaved:")
print("data/processed/engineered_svm_cv_results.csv")
print("data/processed/engineered_svm_summary.csv")