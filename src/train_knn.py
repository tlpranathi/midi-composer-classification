from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "midi_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

# load features
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["composer", "filename"])
y = df["composer"]

# use the same train/test split as the SVM experiments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y,)

# scale features because KNN relies on distances
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

# tune the number of neighbors and distance weighting
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean distance
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)

search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True,)

print("Tuning KNN using 5-fold cross-validation...")
search.fit(X_train, y_train)

print("\nBest parameters:")
print(search.best_params_)

print(f"\nBest cross-validation macro-F1: {search.best_score_:.3f}")

# evaluate once on the held-out test set
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

print("\nTuned KNN test results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# save the results without overwriting SVM results
pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT_DIR / "knn_tuning_cv_results.csv",
    index=False,
)

pd.DataFrame([{
    "model": "Tuned KNN",
    "best_params": str(search.best_params_),
    "cv_macro_f1": search.best_score_,
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
}]).to_csv(
    OUTPUT_DIR / "knn_tuning_summary.csv",
    index=False,
)

print("\nSaved CV results to knn_tuning_cv_results.csv")
print("Saved summary to knn_tuning_summary.csv")