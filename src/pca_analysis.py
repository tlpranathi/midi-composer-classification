from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "midi_features_vertical.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["composer", "filename"])
y = df["composer"]


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y,)

X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,)

# fit preprocessing ONLY on training data.
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_dev_imputed = imputer.transform(X_dev)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_dev_scaled = scaler.transform(X_dev_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


pca = PCA(n_components=2, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)

explained_variance = pca.explained_variance_ratio_

print("=" * 60)
print("PCA ANALYSIS")
print("=" * 60)

print(f"Original number of features: {X.shape[1]}")
print(f"PCA components: 2")

print(f"PC1 explained variance: "f"{explained_variance[0]:.3f}")

print(f"PC2 explained variance: "f"{explained_variance[1]:.3f}")

print(f"Total explained variance: "f"{explained_variance.sum():.3f}")

pca_df = pd.DataFrame(
    {
        "PC1": X_train_pca[:, 0],
        "PC2": X_train_pca[:, 1],
        "composer": y_train.values,
    }
)

plt.figure(figsize=(12, 8))

for composer in sorted(pca_df["composer"].unique()):
    subset = pca_df[pca_df["composer"] == composer]
    plt.scatter(subset["PC1"], subset["PC2"], label=composer, alpha=0.75, s=45,)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of 78 MIDI Features")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",)

plt.tight_layout()

output_path = OUTPUT_DIR / "pca_2d_composers.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight",)

plt.close()

print("\nSaved PCA plot to:")
print(output_path)