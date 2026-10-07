from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "midi_features_vertical.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

# remove identifiers
X = df.drop(columns=["composer", "filename"])
composer = df["composer"]

era_mapping = {
    "albeniz": "Romantic",
    "beeth": "Romantic",
    "chopin": "Romantic",
    "grieg": "Romantic",
    "haydn": "Classical",
    "liszt": "Romantic",
    "mendelssohn": "Romantic",
    "mozart": "Classical",
    "schubert": "Romantic",
    "schumann": "Romantic",
    "tschai": "Romantic",
}

y = composer.map(era_mapping)

# check that every composer received an era
if y.isna().any():
    missing_composers = composer[y.isna()].unique()
    raise ValueError(f"Missing era mapping for: {missing_composers}")

# rhythm-related features
rhythm_features = ["mean_note_duration", "std_note_duration", "median_note_duration", "note_density",]

# melodic + vertical interval features
interval_features = ["mean_interval", "std_interval",]

interval_features += [column for column in X.columns if column.startswith("melodic_interval_")]

interval_features += [column for column in X.columns if column.startswith("vertical_interval_")]


# texture-related features
texture_features = ["instrument_count", "polyphony_ratio",]


# test all features as a reference
feature_groups = {"Rhythm": rhythm_features, "Intervals": interval_features, "Texture": texture_features, "All Features": list(X.columns),}


print("=" * 60)
print("ERA CLASSIFICATION")
print("=" * 60)

print("\nEra distribution:")
print(y.value_counts())

print("\nFeature groups:")

for group_name, features in feature_groups.items():
    print(f"{group_name}: {len(features)} features")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y,)

X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,)


print("\nData split:")
print(f"Training: {len(X_train)}")
print(f"Development: {len(X_dev)}")
print(f"Test: {len(X_test)}")

results = []

for group_name, features in feature_groups.items():

    print("\n" + "=" * 60)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 60)

    X_train_group = X_train[features]
    X_dev_group = X_dev[features]

    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced", random_state=42,),)

    model.fit(X_train_group, y_train,)

    predictions = model.predict(X_dev_group)

    accuracy = accuracy_score(y_dev, predictions,)

    macro_f1 = f1_score(y_dev, predictions, average="macro", zero_division=0,)

    results.append(
        {
            "feature_group": group_name,
            "number_of_features": len(features),
            "dev_accuracy": accuracy,
            "dev_macro_f1": macro_f1,
        }
    )

    print(f"Development Accuracy: {accuracy:.3f}")

    print(f"Development Macro-F1: {macro_f1:.3f}")

results_df = pd.DataFrame(results)

results_path = (OUTPUT_DIR / "era_classification_results.csv")

results_df.to_csv(results_path, index=False,)

print("\n" + "=" * 60)
print("ERA CLASSIFICATION SUMMARY")
print("=" * 60)

print(results_df.to_string(index=False))

plt.figure(figsize=(9, 6))

plt.bar(results_df["feature_group"], results_df["dev_accuracy"],)

plt.ylabel("Development Accuracy")
plt.xlabel("Feature Group")
plt.title("Era Classification Accuracy by Feature Group")

plt.xticks(rotation=20)

plt.tight_layout()

accuracy_plot = (OUTPUT_DIR / "era_classification_accuracy.png")

plt.savefig(accuracy_plot, dpi=200,)

plt.close()

plt.figure(figsize=(9, 6))

plt.bar(results_df["feature_group"], results_df["dev_macro_f1"],)

plt.ylabel("Development Macro-F1")
plt.xlabel("Feature Group")
plt.title("Era Classification Macro-F1 by Feature Group")

plt.xticks(rotation=20)

plt.tight_layout()

f1_plot = (OUTPUT_DIR / "era_classification_macro_f1.png")

plt.savefig(f1_plot, dpi=200,)

plt.close()

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)

print("Results:")
print(results_path)

print("\nAccuracy plot:")
print(accuracy_plot)

print("\nMacro-F1 plot:")
print(f1_plot)