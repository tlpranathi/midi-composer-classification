from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "midi_features_vertical.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["composer", "filename"])
y = df["composer"]


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y,)

X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,)


print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Development samples: {len(X_dev)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")

# these are the best SVM parameters found earlier
svm_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced", random_state=42,),)

print("\nTraining RBF SVM...")

svm_model.fit(X_train, y_train,)

baseline_predictions = svm_model.predict(X_dev)

baseline_macro_f1 = f1_score(y_dev, baseline_predictions, average="macro", zero_division=0,)

print(f"Baseline development Macro-F1: " f"{baseline_macro_f1:.3f}")

print("\nCalculating permutation importance...")
print("This may take a little while...")

importance = permutation_importance(svm_model, X_dev, y_dev, scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1,)

importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }
)

importance_df = importance_df.sort_values("importance_mean", ascending=False,)

print("\n" + "=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)

print(importance_df.head(20).to_string(index=False))

importance_path = (OUTPUT_DIR / "feature_importance.csv")

importance_df.to_csv(importance_path, index=False,)

top_features = importance_df.head(20).copy()

# reverse order so the most important feature appears at top
top_features = top_features.iloc[::-1]

plt.figure(figsize=(10, 8))

plt.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"],)

plt.xlabel("Mean decrease in Macro-F1")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importances - RBF SVM")

plt.tight_layout()

plot_path = (OUTPUT_DIR / "feature_importance_top20.png")

plt.savefig(plot_path, dpi=200,)

plt.close()

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)

print("Full feature importance table:")
print(importance_path)

print("\nFeature importance plot:")
print(plot_path)