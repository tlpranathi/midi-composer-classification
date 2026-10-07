from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

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
print("PCA + SVM ANALYSIS")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Development samples: {len(X_dev)}")
print(f"Test samples: {len(X_test)}")
print(f"Original features: {X.shape[1]}")


# fit preprocessing ONLY on training data
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_dev_imputed = imputer.transform(X_dev)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_dev_scaled = scaler.transform(X_dev_imputed)

component_counts = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 78,]


results = []

for n_components in component_counts:

    print(f"\nTesting {n_components} PCA components...")

    # Fit PCA ONLY on training data
    pca = PCA(n_components=n_components, random_state=42,)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_dev_pca = pca.transform(X_dev_scaled)

    # Tuned SVM parameters from our earlier experiment
    svm = SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced", random_state=42,)

    # train on training set
    svm.fit(X_train_pca, y_train)

    # evaluate on development set
    predictions = svm.predict(X_dev_pca)

    accuracy = accuracy_score(y_dev, predictions,)

    macro_f1 = f1_score(y_dev, predictions, average="macro", zero_division=0,)

    explained_variance = pca.explained_variance_ratio_.sum()

    results.append(
        {
            "n_components": n_components,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "explained_variance": explained_variance,
        }
    )

    print(
        f"Accuracy: {accuracy:.3f} | "
        f"Macro-F1: {macro_f1:.3f} | "
        f"Explained variance: {explained_variance:.3f}"
    )


results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "pca_svm_results.csv"

results_df.to_csv(results_path, index=False,)


best_row = results_df.loc[results_df["macro_f1"].idxmax()]

print("\n" + "=" * 60)
print("BEST PCA CONFIGURATION")
print("=" * 60)

print(f"Components: {int(best_row['n_components'])}")

print(f"Dev Accuracy: {best_row['accuracy']:.3f}")

print(f"Dev Macro-F1: {best_row['macro_f1']:.3f}")

print(f"Explained variance: "f"{best_row['explained_variance']:.3f}")

plt.figure(figsize=(10, 6))

plt.plot(results_df["n_components"], results_df["accuracy"], marker="o",)

plt.xlabel("Number of PCA Components")
plt.ylabel("Development Accuracy")
plt.title("SVM Accuracy vs Number of PCA Components")

plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_plot_path = (OUTPUT_DIR / "pca_svm_accuracy_vs_components.png")

plt.savefig(accuracy_plot_path, dpi=200,)

plt.close()

plt.figure(figsize=(10, 6))

plt.plot(results_df["n_components"], results_df["macro_f1"], marker="o",)

plt.xlabel("Number of PCA Components")
plt.ylabel("Development Macro-F1")
plt.title("SVM Macro-F1 vs Number of PCA Components")

plt.grid(True, alpha=0.3)
plt.tight_layout()

f1_plot_path = (OUTPUT_DIR / "pca_svm_macro_f1_vs_components.png")

plt.savefig(f1_plot_path, dpi=200,)

plt.close()

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)

print("Results:")
print(results_path)

print("\nAccuracy plot:")
print(accuracy_plot_path)

print("\nMacro-F1 plot:")
print(f1_plot_path)