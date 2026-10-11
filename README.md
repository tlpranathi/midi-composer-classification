# MIDI Composer Classification

Can symbolic MIDI data tell you who wrote a piece of classical piano music? This project answers that question with two complementary approaches on the same data and the same train/dev/test split:

1. **Feature-based machine learning.** 78 hand-engineered musical features are extracted from each MIDI file and fed to classic scikit-learn classifiers.
2. **Deep learning.** The same features go into a small fully connected network, and the raw music, converted to piano-roll images, goes into a convolutional network, with and without data augmentation.

A Streamlit app lets you pick a piece or upload your own MIDI file and see the predicted composer with probabilities.

## Results at a glance

| Task | Best setup | Result |
| --- | --- | --- |
| Composer classification (11 classes) | Fully connected NN, 78 features | **51.0% accuracy, 0.502 macro-F1** on a held-out test set of 49 pieces (25 correct) |
| Composer classification (11 classes) | Logistic Regression, 78 features | 46.9% accuracy, 0.451 macro-F1 (23 of 49 correct) |
| Composer classification (11 classes) | CNN on piano rolls, with augmentation | 26.5% accuracy, 0.293 macro-F1 (13 of 49 correct) |
| Era classification (Classical vs. Romantic) | RBF SVM, 78 features | 98.0% accuracy, 0.964 macro-F1 on the development set (see the baseline caveat below) |

For context, guessing uniformly among 11 composers gives about 9%, and always predicting the largest class (Chopin) gives about 18% on the test set. Differences of a few pieces between models are within the noise of a 49-piece test set; see [Limitations](#limitations).

## Contents

- [Dataset](#dataset)
- [Approach overview](#approach-overview)
- [Features](#features)
- [Piano-roll representation](#piano-roll-representation)
- [Models](#models)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Running the demo app](#running-the-demo-app)
- [Evaluation setup](#evaluation-setup)
- [Results](#results)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Conclusion](#conclusion)
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

## Approach overview

```text
                         +--> 78 numerical features --> Logistic Regression, SVM, KNN, LDA,
                         |    (extract_features.py)      Gradient Boosting (scikit-learn)
MIDI files --------------+                          \
                         |                           +--> Fully connected NN (PyTorch)
                         |
                         +--> 64 x 1536 piano rolls ----> CNN, +/- augmentation (PyTorch)
                              (piano_roll_conversion.py)

All models use the same stratified 60/20/20 split (seed 42) and are compared on the same 49 test pieces.
Predicted composer + probabilities are served by the Streamlit app (src/app.py).
```

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

## Piano-roll representation

For the convolutional network, each MIDI file is converted to an image-like matrix by `src/piano_roll_conversion.py`:

- **Shape:** 64 pitches by 1536 time steps.
- **Time resolution:** 10 Hz, so 1536 steps cover 153.6 seconds. Longer pieces are cropped to the first 153.6 s and shorter ones are zero-padded.
- **Pitch range:** MIDI notes 24 to 87 (C1 to D#6). Notes outside this range are dropped.
- **Cell value:** note velocity (0 to 127), scaled to 0-1 for the network.
- All non-drum instruments are merged into one roll.

The rolls are saved to `data/processed/piano_rolls/piano_rolls.npz`, in the same row order as the feature CSV so the two representations line up piece by piece. This file is regenerated by the script and is not committed to git.

## Models

### Classical machine learning (78 features)

Logistic Regression, RBF SVM (`C=10`, `gamma=0.01`), Gradient Boosting, KNN (`k=7`, distance-weighted, Euclidean), and LDA. Pipelines use median imputation and standard scaling where appropriate. Logistic Regression and SVM use balanced class weights. The model with the best development macro-F1 is refit on train + development and scored once on the test set.

### Fully connected neural network (78 features)

`src/train_fcnn.py` trains a PyTorch network with one hidden layer (128 units, ReLU, dropout 0.3) on the same 78 features. The imputer and scaler are fit on the training set only. Training uses class-weighted cross-entropy, AdamW (learning rate 1e-3, weight decay 1e-3), up to 400 epochs, and early stopping on development macro-F1 (patience 60). The best development checkpoint is evaluated once on the test set. Unlike the classical final model, it is trained on the training set only, not train + development.

### Convolutional neural network (piano rolls)

`src/train_cnn.py` trains a CNN on the 64 x 1536 piano rolls: five blocks of 3x3 convolution, batch normalization, ReLU, and 2x2 max-pooling with channels 16, 32, 64, 64, and 128, followed by global average pooling, dropout 0.5, and a linear classifier. It uses class-weighted cross-entropy, AdamW (learning rate 1e-3, weight decay 1e-3), batch size 16, up to 60 epochs, and early stopping on development macro-F1 (patience 15). This is a deliberately small version of the deeper architectures used in the literature, sized for a dataset of 245 pieces.

Run with `--augment`, each training roll is randomly modified on the fly:

| Augmentation | Range |
| --- | --- |
| Pitch shift (transposition) | up to ±6 semitones, zero-filled |
| Time shift | up to ±15 seconds (150 steps), zero-filled |
| Masking | a random time segment of up to 20 seconds is zeroed |

## Project structure

```text
data/
  midi/                  # raw MIDI files, one folder per composer
  processed/             # features, results, plots, and reports
    archive/             # earlier experiments and feature sets

models/                  # trained PyTorch weights and preprocessing used by the demo

src/
  audit_dataset.py          # validity, note count, and duplicate checks
  extract_features.py       # builds the 78-feature representation
  compare_features.py       # compares feature sets as features are added
  tune_svm.py               # SVM hyperparameter search
  tune_engineered_svm.py    # SVM search on engineered features
  train_knn.py              # KNN tuning
  train_lda.py              # LDA baseline
  train_models.py           # final classical model comparison and test evaluation
  pca_analysis.py           # 2D PCA visualization
  pca_svm_analysis.py       # SVM performance vs. number of PCA components
  feature_importance.py     # permutation feature importance
  era_classification.py     # Classical vs. Romantic classification
  piano_roll_conversion.py  # MIDI -> 64 x 1536 piano rolls
  dl_common.py              # shared split, label encoding, metrics, results logging
  train_fcnn.py             # fully connected NN on the 78 features
  train_cnn.py              # CNN on piano rolls, with or without augmentation
  app.py                    # Streamlit demo
```

## Quick start

```bash
git clone https://github.com/tlpranathi/midi-composer-classification
cd midi-composer-classification

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn matplotlib pretty_midi joblib torch streamlit
```

### 1. Feature-based pipeline

Run from the repository root:

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

### 2. Deep learning pipeline

These steps need the feature CSV from step 1 (it is already committed, so you can skip step 1 if you only want the neural models). Run them in this order:

```bash
python src/piano_roll_conversion.py    # creates data/processed/piano_rolls/piano_rolls.npz
python src/train_fcnn.py               # fully connected NN
python src/train_cnn.py                # CNN, no augmentation
python src/train_cnn.py --augment      # CNN, with augmentation
```

Training runs on a GPU or Apple Silicon (MPS) if one is available and on the CPU otherwise. The two CNN runs are the slowest part.

### Outputs

All outputs are written to `data/processed/` (results) and `models/` (weights):

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
| `piano_roll_conversion.py` | `piano_rolls/piano_rolls.npz` (not committed) |
| `train_fcnn.py` | `models/fcnn.pt`, `models/fcnn_prep.joblib`, `fcnn_classification_report.txt`, a row in `dl_results.csv` |
| `train_cnn.py` | `models/cnn_noaug.pt` or `models/cnn_aug.pt`, `cnn_*_classification_report.txt`, `cnn_*_confusion_matrix.png`, a row in `dl_results.csv` |

All random seeds are fixed at 42, so reruns should reproduce the same splits. Small numerical differences in the neural network results can still appear between CPU, GPU, and Apple Silicon.

## Running the demo app

The trained weights are included in `models/`, so the app works without retraining.

```bash
python -m streamlit run src/app.py
```

Open the local URL it prints (usually <http://localhost:8501>). Choose how to supply a piece:

- **Pick from dataset:** select a composer folder and a file. The app shows the predictions of both the CNN and the fully connected NN. If the piece was in the training set, the prediction is optimistic.
- **Upload a file:** upload any `.mid` or `.midi` file. The app shows the CNN prediction only, because the fully connected NN needs the 78 stored features, which are precomputed for dataset files only.

For each model the app shows the predicted composer and a bar chart of the class probabilities, plus the piano roll of the piece. The CNN uses the augmented model if `models/cnn_aug.pt` exists and falls back to the non-augmented one otherwise.

## Evaluation setup

The final composer experiment uses a **stratified 60/20/20 split**, shared by every model in the project:

| Split | Pieces | Purpose |
| --- | ---: | --- |
| Training | 147 | Fit candidate models |
| Development | 49 | Compare models, tune, and early-stop |
| Test | 49 | Evaluated once per model, after model selection |

The classical final model (highest development macro-F1) is refit on train + development (196 pieces) before the single test evaluation. The neural networks are trained on the training set only, with early stopping on the development set. Macro-F1 is the selection metric because it weights every composer equally despite the class imbalance.

## Results

### Composer classification: all models

Development-set comparison of the classical models:

| Model | Dev accuracy | Dev macro-F1 |
| --- | ---: | ---: |
| **Logistic Regression** | 0.469 | **0.479** |
| RBF SVM | **0.531** | 0.402 |
| KNN | 0.429 | 0.353 |
| Gradient Boosting | 0.388 | 0.340 |
| LDA | 0.327 | 0.280 |

The SVM has the highest accuracy, but Logistic Regression has the best macro-F1 and was selected. On 49 development pieces the gap between these models is small enough that the choice should not be over-read.

Final results on the held-out test set:

| Model | Input | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Fully connected NN | 78 features | 0.592 | 0.547 | **0.510** (25/49) | **0.502** |
| Logistic Regression | 78 features | 0.469 | 0.479 | 0.469 (23/49) | 0.451 |
| CNN, augmented | piano rolls | 0.286 | 0.241 | 0.265 (13/49) | 0.293 |
| CNN, no augmentation | piano rolls | 0.286 | 0.255 | 0.184 (9/49) | 0.133 |

What the numbers do and do not say:

- **Features beat raw piano rolls here.** With 147 training pieces, a CNN has too little data to learn stylistic patterns from scratch, while the engineered features already encode musical structure such as intervals and pitch-class usage.
- **The CNN without augmentation is at the majority-class baseline.** Always predicting Chopin scores 9/49 (18.4%), which equals its test accuracy, so it learned little that generalizes.
- **Augmentation helped the CNN, but treat the size of the gain with care.** Test accuracy went from 9 to 13 correct pieces and macro-F1 from 0.133 to 0.293, while development accuracy was identical (0.286) for both runs. Each run uses a single seed and 49 test pieces.
- **The fully connected NN and Logistic Regression differ by 2 pieces** (25 vs. 23 correct), which is not a reliable ranking. They are also not trained identically: the NN uses the training set only with early stopping on the development set, whereas the final Logistic Regression is refit on train + development.

### Per-composer results: Logistic Regression

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

Liszt and Haydn are recognized best, while Grieg is never identified correctly. The confusion matrix is in `data/processed/confusion_matrix_final_logistic_regression.png`. Per-composer reports and confusion matrices for the neural networks are in `data/processed/fcnn_classification_report.txt` and `data/processed/cnn_*_classification_report.txt` / `cnn_*_confusion_matrix.png`. For the fully connected NN, Haydn is recognized perfectly and Mozart, Schubert, and Albéniz are also well recognized, while Grieg is again never identified.

### Feature set ablation

An exploratory experiment with the RBF SVM adds feature groups step by step. It used an earlier 80/20 holdout split, **not** the final 60/20/20 evaluation, so these numbers are not comparable to the test results above.

| Feature set | Features | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original | 20 | 0.408 | 0.388 |
| Engineered | 28 | 0.510 | 0.462 |
| + Melodic interval histogram | 53 | 0.551 | 0.511 |
| + Vertical interval histogram | 78 | 0.571 | 0.545 |

Accuracy rises with every added group, which suggests that interval information helps. The whole range from 20 to 78 features spans 8 pieces out of 49 (20 vs. 28 correct), so the trend is encouraging but not statistically conclusive.

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

- **Small dataset and test set.** With 245 pieces and 49 test pieces (as few as 3 per composer), one prediction moves accuracy by about 2 points. Neural network results come from a single seed.
- **Possible leakage between related pieces.** Splits are made per MIDI file, not per composition, so movements of the same work can land in both training and test sets and make the task look easier than it is. This affects every model, including the neural networks.
- **Class imbalance.** Class weights and macro-F1 reduce, but do not remove, the influence of the larger composers.
- **Approximate melodic intervals.** They are computed from all notes sorted by start time, without separating voices or hands, so they describe overall note-to-note pitch movement and not a true melodic line.
- **Vertical interval sign.** The sign of a vertical interval depends on note ordering, so individual signed bins should not be read as "ascending" or "descending" harmony.
- **Piano-roll truncation.** Pieces are cropped to the first 153.6 seconds, and notes outside MIDI 24-87 are dropped.
- **CNN limited by data.** A CNN trained from scratch on 147 pieces underperforms feature-based models, and its augmentation gains are measured on a very small test set.
- **Simplified era labels.** The Classical vs. Romantic split is a convenience grouping, and it is heavily skewed toward Romantic.
- **Model selection on a small dev set.** Tuning, model choice, and early stopping used the same 49 development pieces, which can overfit that set.
- **Demo predictions on training data.** For dataset files that were in the training set, the app's predictions are optimistic. Uploaded files get the CNN prediction only.

## Future work

- Split by composition (`GroupKFold` or similar) and use repeated cross-validation and multiple seeds for more reliable estimates
- Extract a proper melodic line (for example, the top voice, or per-hand tracks)
- Add rhythm, tempo, and key-normalized pitch-class features
- Report a majority-class baseline next to every result
- Collect more data or use pretrained or self-supervised music models, and try sequence models on piano rolls or note tokens
- Compute the 78 features for uploaded files inside the app so the fully connected NN also works on uploads

## Conclusion

Hand-engineered symbolic features carry real stylistic information: the best models identify the composer correctly about 47-51% of the time across 11 classes, several times better than chance, and the broader Classical vs. Romantic split is much easier than telling individual composers apart. Interval-based features contribute the most. A CNN on raw piano rolls does not match the feature-based models at this dataset size, although augmentation narrows the gap somewhat. The small, imbalanced dataset and the possibility of related pieces sharing splits mean the exact numbers should be treated as estimates.

## Data source

Classical Music MIDI dataset by Soumik Rakshit, Kaggle: <https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi>
