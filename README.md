# MIDI Composer Classification

Can symbolic MIDI data tell you who wrote a piece of classical piano music? This project extracts 78 hand-engineered musical features from MIDI files, compares classic scikit-learn classifiers, and measures which kinds of features carry the most stylistic signal.

## Results at a glance

| Task | Best setup | Result |
| --- | --- | --- |
| Composer classification (11 classes) | Logistic Regression, 78 features | **46.9% accuracy, 0.451 macro-F1** on a held-out test set of 49 pieces (23 correct) |
| Era classification (Classical vs. Romantic) | RBF SVM, 78 features | 98.0% accuracy, 0.964 macro-F1 on the development set (see the baseline caveat below) |

For context, guessing uniformly among 11 composers gives about 9%, and always predicting the largest class (Chopin) gives about 20%.

## Contents

- [Dataset](#dataset)
- [Features](#features)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Evaluation setup](#evaluation-setup)
- [Results](#results)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Data source](#data-source)

## Dataset

The data is the **Classical Music MIDI** dataset by Soumik Rakshit on Kaggle: <https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi>

The project uses 245 MIDI files from 11 composers, stored as `data/midi/<composer>/`. `src/audit_dataset.py` confirmed that every file parses, contains notes, and is not a byte-identical duplicate. The classes are imbalanced, ranging from 12 to 48 pieces.

| Folder | Composer | Pieces |
| --- | --- | ---: |
| `chopin` | Chopin | 48 |
| `beeth` | Beethoven | 29 |
| `schubert` | Schubert | 29 |
| `schumann` | Schumann | 24 |
| `haydn` | Haydn | 21 |
| `mozart` | Mozart | 21 |
| `grieg` | Grieg | 16 |
| `liszt` | Liszt | 16 |
| `mendelssohn` | Mendelssohn | 15 |
| `albeniz` | Albéniz | 14 |
| `tschai` | Tchaikovsky | 12 |

## Features

Each piece is described by **78 numerical features** computed with [`pretty_midi`](https://github.com/craffel/pretty-midi) in `src/extract_features.py`.

| Group | Count | What it captures |
| --- | ---: | --- |
| General | 8 | Mean and std of pitch, mean and std of note duration, note density, mean and std of interval, instrument count |
| Pitch and duration | 5 | Min, max, range, and median pitch; median note duration |
| Tonal and texture | 3 | Pitch-class entropy, chromatic note ratio (share of notes outside the best-fitting major or minor scale), polyphony ratio |
| Pitch-class distribution | 12 | Share of notes on each of the 12 pitch classes |
| Melodic interval histogram | 25 | Normalized histogram of intervals from -12 to +12 semitones between consecutive notes; larger intervals are clipped into the edge bins |
| Vertical interval histogram | 25 | Normalized histogram of intervals from -12 to +12 semitones between simultaneously sounding notes, excluding unisons |

## Project structure

```text
data/
  midi/                  # raw MIDI files, one folder per composer
  processed/             # features, results, plots, and reports
    archive/             # earlier experiments and feature sets

src/
  audit_dataset.py       # validity, note count, and duplicate checks
  extract_features.py    # builds the 78-feature representation
  compare_features.py    # compares feature sets as features are added
  tune_svm.py            # SVM hyperparameter search
  tune_engineered_svm.py # SVM search on engineered features
  train_knn.py           # KNN tuning
  train_lda.py           # LDA baseline
  train_models.py        # final model comparison and test evaluation
  pca_analysis.py        # 2D PCA visualization
  pca_svm_analysis.py    # SVM performance vs. number of PCA components
  feature_importance.py  # permutation feature importance
  era_classification.py  # Classical vs. Romantic classification
```

## Quick start

```bash
git clone https://github.com/tlpranathi/midi-composer-classification
cd midi-composer-classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn matplotlib pretty_midi
```

Run the main pipeline from the repository root:

```bash
python src/audit_dataset.py        # optional dataset sanity check
python src/extract_features.py     # creates the 78-feature CSV
python src/train_models.py         # compares models, evaluates the final one
```

The analysis scripts read the feature CSV and can be run in any order afterwards:

```bash
python src/compare_features.py
python src/pca_analysis.py
python src/pca_svm_analysis.py
python src/feature_importance.py
python src/era_classification.py
```

All outputs are written to `data/processed/`:

| Script | Main outputs |
| --- | --- |
| `audit_dataset.py` | `dataset_audit.csv` |
| `extract_features.py` | `midi_features_vertical.csv` |
| `train_models.py` | `model_comparison_dev.csv`, `final_model_results.csv`, `final_classification_report.txt`, `confusion_matrix_final_logistic_regression.png` |
| `compare_features.py` | `feature_comparison.csv` |
| `pca_analysis.py` | `pca_2d_composers.png` |
| `pca_svm_analysis.py` | `pca_svm_results.csv` and two accuracy/macro-F1 plots |
| `feature_importance.py` | `feature_importance.csv`, `feature_importance_top20.png` |
| `era_classification.py` | `era_classification_results.csv` and two accuracy/macro-F1 plots |

All random seeds are fixed at 42, so reruns should reproduce the same splits and results.

## Evaluation setup

The final composer experiment uses a **stratified 60/20/20 split**:

| Split | Pieces | Purpose |
| --- | ---: | --- |
| Training | 147 | Fit candidate models |
| Development | 49 | Compare models and pick the best |
| Test | 49 | Evaluated once, after model selection |

The selected model (highest development macro-F1) is refit on train + development (196 pieces) and scored a single time on the test set.

Pipelines use median imputation and standard scaling where appropriate. Logistic Regression and SVM use balanced class weights. The compared models are Logistic Regression, RBF SVM (`C=10`, `gamma=0.01`), Gradient Boosting, KNN (`k=7`, distance-weighted, Euclidean), and LDA. Macro-F1 is the selection metric because it weights every composer equally despite the class imbalance.

## Results

### Composer classification

Development-set comparison:

| Model | Dev accuracy | Dev macro-F1 |
| --- | ---: | ---: |
| **Logistic Regression** | 0.469 | **0.479** |
| RBF SVM | **0.531** | 0.402 |
| KNN | 0.429 | 0.353 |
| Gradient Boosting | 0.388 | 0.340 |
| LDA | 0.327 | 0.280 |

The SVM has the highest accuracy, but Logistic Regression has the best macro-F1 and was selected. On 49 development pieces the gap between these models is small enough that the choice should not be over-read.

Final test result for Logistic Regression: **46.9% accuracy, 0.451 macro-F1** (23 of 49 correct).

| Composer | Precision | Recall | F1 | Test pieces |
| --- | ---: | ---: | ---: | ---: |
| Liszt | 1.00 | 0.67 | 0.80 | 3 |
| Haydn | 0.75 | 0.75 | 0.75 | 4 |
| Chopin | 0.56 | 0.56 | 0.56 | 9 |
| Schubert | 0.40 | 0.67 | 0.50 | 6 |
| Schumann | 0.67 | 0.40 | 0.50 | 5 |
| Mozart | 0.50 | 0.50 | 0.50 | 4 |
| Tchaikovsky | 0.40 | 0.67 | 0.50 | 3 |
| Albéniz | 0.50 | 0.33 | 0.40 | 3 |
| Mendelssohn | 0.20 | 0.33 | 0.25 | 3 |
| Beethoven | 0.25 | 0.17 | 0.20 | 6 |
| Grieg | 0.00 | 0.00 | 0.00 | 3 |

Liszt and Haydn are recognized best, while Grieg is never identified correctly. The confusion matrix is in `data/processed/confusion_matrix_final_logistic_regression.png`.

### Feature set ablation

An exploratory experiment with the RBF SVM adds feature groups step by step. It used the earlier 80/20 holdout split, **not** the final 60/20/20 evaluation, so these numbers are not comparable to the 46.9% test result above.

| Feature set | Features | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original | 20 | 0.408 | 0.388 |
| Engineered | 28 | 0.510 | 0.462 |
| + Melodic interval histogram | 53 | 0.551 | 0.511 |
| + Vertical interval histogram | 78 | 0.571 | 0.545 |

Accuracy rises with every added group, which suggests that interval information helps. Keep in mind that the whole range from 20 to 78 features spans 8 pieces out of 49 (20 vs. 28 correct), so the trend is encouraging but not statistically conclusive.

### Era classification

This experiment asks whether the features separate **Classical** pieces (Haydn, Mozart) from **Romantic** pieces (all other composers). The grouping is a simplified project-level choice, not a full historical classification.

Development-set results with the RBF SVM:

| Feature group | Features | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Rhythm | 4 | 0.816 | 0.758 |
| Texture | 2 | 0.694 | 0.617 |
| Intervals | 52 | 0.939 | 0.893 |
| All features | 78 | 0.980 | 0.964 |

**Baseline caveat:** about 83% of the dataset (203 of 245 pieces) is Romantic, so always predicting "Romantic" would score roughly 83% accuracy. Macro-F1 is the fairer number here. Even so, the rhythm-only and texture-only accuracies sit at or below that baseline, whereas the interval features and the full set clearly exceed it. Results are on the development set, not a held-out test set.

### PCA analysis

PCA was used to study the dimensionality of the feature space. The first two components explain only about **23.6%** of the variance, so a 2D plot (`pca_2d_composers.png`) cannot separate composers cleanly.

| PCA components | Variance explained | Dev accuracy | Dev macro-F1 |
| ---: | ---: | ---: | ---: |
| 2 | 23.6% | 0.204 | 0.234 |
| 30 | 89.4% | 0.551 | 0.479 |
| 78 (all) | 100% | 0.531 | 0.402 |

Performance peaks around 30 components and does not keep improving beyond that, so the useful information is spread over many dimensions and a fair amount of the 78 features is redundant. The best component count was chosen on the development set, so treat it as a trend rather than a tuned result.

### Feature importance

Permutation importance for the SVM (development set, macro-F1) ranks a few vertical-interval bins (-6, -2, +11) and pitch-register features (mean pitch, pitch standard deviation) highest. The standard deviations across repeats are about as large as the means, so use the ranking as a loose guide, not a finding. Full results are in `data/processed/feature_importance.csv`.

## Limitations

- **Small dataset and test set.** With 245 pieces and 49 test pieces (as few as 3 per composer), one prediction moves accuracy by about 2 points.
- **Possible leakage between related pieces.** Splits are made per MIDI file, not per composition, so movements of the same work can land in both training and test sets and make the task look easier than it is.
- **Class imbalance.** Class weights and macro-F1 reduce, but do not remove, the influence of the larger composers.
- **Approximate melodic intervals.** They are computed from all notes sorted by start time, without separating voices or hands, so they describe overall note-to-note pitch movement and not a true melodic line.
- **Vertical interval sign.** The sign of a vertical interval depends on note ordering, so individual signed bins should not be read as "ascending" or "descending" harmony.
- **Simplified era labels.** The Classical vs. Romantic split is a convenience grouping, and it is heavily skewed toward Romantic.
- **Model selection on a small dev set.** Tuning and model choice used the same 49 development pieces, which can overfit that set.

## Future work

- Split by composition (`GroupKFold` or similar) and use repeated cross-validation for more reliable estimates
- Extract a proper melodic line (for example, the top voice, or per-hand tracks)
- Add rhythm, tempo, and key-normalized pitch-class features
- Report a majority-class baseline next to every result
- Try sequence models on piano rolls or note tokens, which would need more data

## Conclusion

Hand-engineered symbolic features carry real stylistic information: the final model identifies the composer correctly about 47% of the time across 11 classes, several times better than chance, and the broader Classical vs. Romantic split is much easier than telling individual composers apart. Interval-based features contribute the most. The small, imbalanced dataset and the possibility of related pieces sharing splits mean the exact numbers should be treated as estimates.
