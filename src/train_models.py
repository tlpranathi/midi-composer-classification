from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay,)



ROOT = Path(__file__).resolve().parent.parent

# use the restored 78-feature dataset
DATA_PATH = ROOT / "data" / "processed" / "midi_features_vertical.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

# filename is an identifier, not a predictive feature.
X = df.drop(columns=["composer", "filename"])
y = df["composer"]

# first split:
# 60% training
# 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y,)

# second split:
# half of the remaining 40% goes to development
# half goes to final test

# therefore:
# 60% train
# 20% dev
# 20% test
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,)


print("=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Development samples: {len(X_dev)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
    ),

    # best SVM parameters found during previous tuning
    "RBF SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced", random_state=42,),
    ),

    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(random_state=42,),
    ),

    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7, weights="distance", p=2,),
    ),

    "LDA": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearDiscriminantAnalysis(),
    ),
}

dev_results = []

print("\n")
print("=" * 60)
print("DEVELOPMENT SET MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining: {name}")

    # fit ONLY on training data
    model.fit(X_train, y_train)

    # evaluate on development data
    dev_predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, dev_predictions)

    macro_f1 = f1_score(y_dev, dev_predictions, average="macro", zero_division=0,)

    dev_results.append(
        {
            "model": name,
            "dev_accuracy": accuracy,
            "dev_macro_f1": macro_f1,
        }
    )

    print(f"Dev Accuracy: {accuracy:.3f}")
    print(f"Dev Macro-F1: {macro_f1:.3f}")


# convert to DataFrame
dev_results_df = pd.DataFrame(dev_results)

# select the model with the highest development macro-F1
best_model_name = dev_results_df.sort_values("dev_macro_f1", ascending=False,).iloc[0]["model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"Selected using development Macro-F1: {best_model_name}")


# save development comparison
dev_results_path = OUTPUT_DIR / "model_comparison_dev.csv"
dev_results_df.to_csv(dev_results_path, index=False,)

print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

# recreate the selected model
best_model = models[best_model_name]

# combine TRAIN + DEV for the final model fitting
X_train_final = pd.concat([X_train, X_dev], axis=0,)

y_train_final = pd.concat([y_train, y_dev], axis=0,)

print(
    f"Final training data: {len(X_train_final)} samples "
    f"(train + development)"
)

# fit final selected model on train + development data
best_model.fit(X_train_final, y_train_final,)

# evaluate ONCE on the untouched test set
test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions,)

test_macro_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=0,)

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Test Macro-F1: {test_macro_f1:.3f}")

print("\nClassification Report:")
test_report = classification_report(y_test, test_predictions, zero_division=0,)

print(test_report)


fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, labels=sorted(y.unique()), xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False,)

ax.set_title(f"Confusion Matrix - Final {best_model_name}")

fig.tight_layout()

safe_name = (best_model_name.lower().replace(" ", "_"))

fig.savefig(OUTPUT_DIR / f"confusion_matrix_final_{safe_name}.png", dpi=200,)

plt.close(fig)

final_results = pd.DataFrame(
    [
        {
            "best_model": best_model_name,
            "dev_accuracy": dev_results_df.loc[
                dev_results_df["model"] == best_model_name,
                "dev_accuracy",
            ].iloc[0],
            "dev_macro_f1": dev_results_df.loc[
                dev_results_df["model"] == best_model_name,
                "dev_macro_f1",
            ].iloc[0],
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "initial_train_samples": len(X_train),
            "dev_samples": len(X_dev),
            "final_train_samples": len(X_train_final),
            "test_samples": len(X_test),
            "number_of_features": X.shape[1],
        }
    ]
)

final_results_path = OUTPUT_DIR / "final_model_results.csv"

final_results.to_csv(final_results_path, index=False,)


# save detailed classification report
report_path = OUTPUT_DIR / "final_classification_report.txt"

with open(report_path, "w", encoding="utf-8",) as f:

    f.write("=" * 60 + "\n")
    f.write("FINAL MODEL RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Number of features: {X.shape[1]}\n")
    f.write(f"Initial training samples: {len(X_train)}\n")
    f.write(f"Development samples: {len(X_dev)}\n")
    f.write(f"Final training samples: {len(X_train_final)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")

    f.write(f"Test Accuracy: {test_accuracy:.3f}\n")
    f.write(f"Test Macro-F1: {test_macro_f1:.3f}\n\n")

    f.write("Classification Report:\n")
    f.write(test_report)

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)

print("Development comparison:")
print(dev_results_path)

print("\nFinal model results:")
print(final_results_path)

print("\nClassification report:")
print(report_path)

print("\nConfusion matrix:")
print(OUTPUT_DIR / f"confusion_matrix_final_{safe_name}.png")