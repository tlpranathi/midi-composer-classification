# MIDI Composer Classification
Machine Learning Mini-Project Assignment

**Project title:** Analysis and Classification of Symbolic Western Classical Music by Composer

## Team and contribution split

| Team member | Responsibilities |
| --- | --- |
| Lakshmi Pranathi | Dataset preparation and auditing, feature extraction and engineering, classical machine-learning models, PCA analysis, feature importance, confusion-matrix analysis, and era classification |
| Marella Likhita Sri Durga | MIDI-to-piano-roll conversion, CNN implementation, data-augmentation experiments, fully connected neural network, and Streamlit demo integration |

Can symbolic MIDI data tell you who wrote a piece of classical music? This project explores that question with two complementary approaches on the same dataset:

1. **Feature-based machine learning:** 78 hand-engineered musical features are extracted from each MIDI file and used by scikit-learn classifiers.
2. **Deep learning:** the same features are passed to a fully connected neural network, while piano-roll representations are passed to a convolutional neural network (CNN), with and without data augmentation.

A Streamlit app lets you select a dataset piece or upload a MIDI file and inspect composer predictions, class probabilities, and a piano-roll visualization.

## Results at a glance

| Task | Model | Evaluation | Result |
| --- | --- | --- | --- |
| Composer classification (11 composers) | Fully connected NN, 78 features | Held-out test: 49 pieces | **51.0% accuracy, 0.502 macro-F1** (25/49 correct) |
| Composer classification (11 composers) | Logistic Regression, 78 features | Held-out test: 49 pieces | 46.9% accuracy, 0.451 macro-F1 (23/49 correct) |
| Composer classification (11 composers) | CNN on piano rolls, with augmentation | Held-out test: 49 pieces | 26.5% accuracy, 0.293 macro-F1 (13/49 correct) |
| Era classification (Classical vs. Romantic) | RBF SVM, 78 features | Development set: 49 pieces | 98.0% accuracy, 0.964 macro-F1* |

\*The era-classification result is on the development set, not a held-out test set. The project-defined labels are imbalanced: 203 of 245 pieces are labeled Romantic, so accuracy alone is not sufficient.

Uniform guessing among 11 composers gives about 9% accuracy. Always predicting the largest class (Chopin) gives 18.4% accuracy on the 49-piece test set. Differences of only a few pieces between models are uncertain at this sample size; see [Limitations](#limitations).

## Contents

- [Dataset](#dataset)
- [Approach overview](#approach-overview)
- [Features](#features)
- [Piano-roll representation](#piano-roll-representation)
- [Models](#models)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipelines](#running-the-pipelines)
- [Running the demo app](#running-the-demo-app)
- [Evaluation setup](#evaluation-setup)
- [Results](#results)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Conclusion](#conclusion)

## Dataset

The project uses the **Classical Music MIDI** dataset by Soumik Rakshit on Kaggle:

<https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi>

The experiments use 245 MIDI files from 11 composers. The expected raw-data layout is:

```text
data/
└── midi/
    ├── albeniz/
    ├── beeth/
    ├── chopin/
    ├── grieg/
    ├── haydn/
    ├── liszt/
    ├── mendelssohn/
    ├── mozart/
    ├── schubert/
    ├── schumann/
    └── tschai/
```

The dataset audit reported that all 245 files parsed successfully and contained notes, with no byte-identical duplicates. This does not rule out different MIDI files representing movements, excerpts, or arrangements from the same composition.

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

**Dataset setup:** Download the dataset from Kaggle and place the composer folders under `data/midi/` as shown above. The raw MIDI files are required to rerun feature extraction and use dataset selection in the app.

## Approach overview

```text
                          +--> 78 numerical features --> Logistic Regression, SVM, KNN,
                          |                              LDA, Gradient Boosting
MIDI files ---------------+                              |
                          |                              +--> Fully connected NN
                          |
                          +--> 64 x 1536 piano rolls ----> CNN (with/without augmentation)

Predicted composer + class probabilities + piano-roll visualization
                              |
                         Streamlit app
```

All composer-classification models use the same stratified 60/20/20 split with random seed 42. The final test set contains 49 pieces.

## Features

Each MIDI file is described by **78 numerical features** extracted using [`pretty_midi`](https://github.com/craffel/pretty-midi) in `src/extract_features.py`.

| Group | Count | What it captures |
| --- | ---: | --- |
| General | 8 | Mean and standard deviation of pitch and note duration, note density, mean and standard deviation of intervals, instrument count |
| Pitch and duration | 5 | Minimum, maximum, range, and median pitch; median note duration |
| Tonal and texture | 3 | Pitch-class entropy, chromatic-note ratio, polyphony ratio |
| Pitch-class distribution | 12 | Share of notes in each of the 12 pitch classes |
| Melodic interval histogram | 25 | Normalized histogram of intervals from -12 to +12 semitones between consecutive notes; larger intervals are clipped into edge bins |
| Vertical interval histogram | 25 | Normalized histogram of intervals from -12 to +12 semitones between simultaneously sounding notes, excluding unisons |

The chromatic-note ratio is a rough heuristic based on major and natural-minor scale templates, not a full key detector. Interval features summarize notes across the MIDI rather than reliably reconstructing individual voices or hands.

## Piano-roll representation

For the CNN, `src/piano_roll_conversion.py` converts each MIDI file into a matrix:

- **Shape:** 64 pitches × 1536 time steps.
- **Time resolution:** 10 Hz; 1536 steps represent 153.6 seconds.
- **Long pieces:** cropped to the first 153.6 seconds; shorter pieces are zero-padded.
- **Pitch range:** MIDI notes 24–87 (C1 to D#6); notes outside this range are dropped.
- **Cell values:** note velocities scaled from 0–127 to 0–1.
- **Instruments:** all non-drum instruments are merged into one piano roll.

The generated file is `data/processed/piano_rolls/piano_rolls.npz`. It is generated by the conversion script and is not committed to Git.

## Models

### Classical machine learning

The project compares Logistic Regression, RBF SVM (`C=10`, `gamma=0.01`), Gradient Boosting, KNN (`k=7`, distance-weighted, Euclidean distance), and LDA. Preprocessing uses median imputation and scaling where appropriate. Logistic Regression and SVM use balanced class weights.

The classical model with the best development macro-F1 is selected, refit on the combined training and development data, and evaluated on the held-out test set.

### Fully connected neural network

`src/train_fcnn.py` trains a PyTorch network with one hidden layer of 128 units, ReLU, and dropout (0.3), using the 78 numerical features. The imputer and scaler are fit on the training set. Training uses class-weighted cross-entropy, AdamW, and early stopping based on development macro-F1. Unlike the final classical model, this network is trained on the training set only, not training plus development.

### Convolutional neural network

`src/train_cnn.py` trains a CNN on the piano rolls. The architecture uses five convolution blocks with channels 16, 32, 64, 64, and 128, followed by global average pooling, dropout, and a classifier. Training uses class-weighted cross-entropy, AdamW, and early stopping based on development macro-F1.

Run `python src/train_cnn.py --augment` to enable on-the-fly augmentation:

| Augmentation | Range |
| --- | --- |
| Pitch transposition | Up to ±6 semitones |
| Time shift | Up to ±15 seconds |
| Time masking | A random segment of up to 20 seconds is zeroed |

This is a small CNN trained from scratch on a limited dataset, so its results should be interpreted in that context.

## Project structure

```text
data/
  midi/                     # Raw MIDI files, one folder per composer
  processed/                # Features, results, plots, and reports
    archive/                # Earlier experiments and feature sets
    piano_rolls/             # Generated piano-roll data (not committed)

models/                     # Trained weights and preprocessing artifacts

src/
  audit_dataset.py
  extract_features.py
  compare_features.py
  tune_svm.py
  tune_engineered_svm.py
  train_knn.py
  train_lda.py
  train_models.py
  pca_analysis.py
  pca_svm_analysis.py
  feature_importance.py
  era_classification.py
  piano_roll_conversion.py
  dl_common.py
  train_fcnn.py
  train_cnn.py
  app.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tlpranathi/midi-composer-classification.git
cd midi-composer-classification
```

### 2. Create and activate a virtual environment

**Linux / macOS / WSL:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies used by the project:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pretty_midi joblib torch streamlit
```

Use a `requirements.txt` file to install the project dependencies consistently across environments.

## Running the pipelines

Run commands from the repository root.

### Feature-based machine learning

```bash
python src/audit_dataset.py
python src/extract_features.py
python src/train_models.py
```

The feature extraction script creates the 78-feature CSV used by the models. The final model script compares classical models on the development set and evaluates the selected model on the test set.

Additional analyses can be run after feature extraction:

```bash
python src/compare_features.py
python src/pca_analysis.py
python src/pca_svm_analysis.py
python src/feature_importance.py
python src/era_classification.py
```

### Deep learning

The deep-learning pipeline requires the feature CSV for the fully connected network and raw MIDI files for piano-roll generation.

```bash
python src/piano_roll_conversion.py
python src/train_fcnn.py
python src/train_cnn.py
python src/train_cnn.py --augment
```

Training can run on supported GPU hardware or on the CPU. CNN training is the slowest part of the pipeline. If trained model artifacts are already included in `models/`, you can try the demo app without retraining.

## Main outputs

Outputs are written to `data/processed/` and `models/`.

| Script | Main outputs |
| --- | --- |
| `audit_dataset.py` | `dataset_audit.csv` |
| `extract_features.py` | `midi_features_vertical.csv` |
| `train_models.py` | `model_comparison_dev.csv`, `final_model_results.csv`, classification report, confusion matrix |
| `compare_features.py` | `feature_comparison.csv` |
| `pca_analysis.py` | `pca_2d_composers.png` |
| `pca_svm_analysis.py` | PCA component results and plots |
| `feature_importance.py` | `feature_importance.csv`, feature-importance plot |
| `era_classification.py` | Era-classification results and plots |
| `piano_roll_conversion.py` | `piano_rolls/piano_rolls.npz` |
| `train_fcnn.py` | FCNN weights, preprocessing artifact, classification report |
| `train_cnn.py` | CNN weights, classification reports, confusion matrices |

The scripts save their outputs under `data/processed/` and `models/`.

## Running the demo app

If the trained weights and preprocessing artifacts are present in `models/`, run:

```bash
python -m streamlit run src/app.py
```

Open <http://localhost:8501> in your browser.

- **Pick from dataset:** select a composer and MIDI file. The app shows CNN and fully connected network predictions for dataset pieces with precomputed features.
- **Upload a MIDI file:** upload a `.mid` or `.midi` file. The app shows the CNN prediction; the fully connected network requires the 78 numerical features, which are precomputed for dataset files only.

The app displays predicted composers, class-probability charts, and a piano-roll visualization. The CNN checkpoint used by the app depends on which model artifacts are available; verify the selection logic in `src/app.py` if you change checkpoints.

## Evaluation setup

The composer-classification experiment uses a stratified 60/20/20 split with random seed 42:

| Split | Pieces | Purpose |
| --- | ---: | --- |
| Training | 147 | Fit candidate models and neural networks |
| Development | 49 | Compare models, tune settings, and early-stop |
| Test | 49 | Final held-out evaluation |

The classical model is refit on the combined training and development sets (196 pieces) before test evaluation. The neural networks use the training set for fitting and the development set for early stopping. Macro-F1 is emphasized because it gives each composer equal weight despite class imbalance.

## Results

### Classical model comparison

Development-set performance:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.469 | **0.479** |
| RBF SVM | **0.531** | 0.402 |
| KNN | 0.429 | 0.353 |
| Gradient Boosting | 0.388 | 0.340 |
| LDA | 0.327 | 0.280 |

The SVM has the highest development accuracy, while Logistic Regression has the highest development macro-F1. Logistic Regression was selected because macro-F1 was the selection metric.

### Held-out test results

| Model | Input | Test accuracy | Test macro-F1 |
| --- | --- | ---: | ---: |
| Fully connected neural network | 78 features | **0.510 (25/49)** | **0.502** |
| Logistic Regression | 78 features | 0.469 (23/49) | 0.451 |
| CNN with augmentation | Piano rolls | 0.265 (13/49) | 0.293 |
| CNN without augmentation | Piano rolls | 0.184 (9/49) | 0.133 |

Interpretation:

- Feature-based approaches perform better than the CNN on this small dataset.
- The CNN without augmentation matches the 18.4% majority-class accuracy baseline on the test set.
- Augmentation improved the CNN's test results from 9 to 13 correct predictions, but this is a small test set and the experiment uses a single seed.
- The fully connected network and Logistic Regression differ by only two correct predictions. They were also trained differently, so these results do not establish that one is definitively better.

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

Several classes have only three test pieces, so individual class scores are rough estimates rather than stable measurements.

### Feature-set ablation

This exploratory experiment used an earlier 80/20 holdout split, not the final 60/20/20 evaluation. Do not compare these scores directly with the held-out test results above.

| Feature set | Features | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original | 20 | 0.408 | 0.388 |
| Engineered | 28 | 0.510 | 0.462 |
| + Melodic interval histogram | 53 | 0.551 | 0.511 |
| + Vertical interval histogram | 78 | 0.571 | 0.545 |

The scores rise as interval features are added, suggesting interval information may be useful. However, the experiment uses only 49 held-out pieces, so the trend is encouraging but not conclusive.

### Era classification

The project labels Haydn and Mozart as **Classical**, and the other nine composers as **Romantic**. This is a simplified project-level grouping, not a complete historical classification.

Development-set results:

| Feature group | Features | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Rhythm | 4 | 0.816 | 0.758 |
| Texture | 2 | 0.694 | 0.617 |
| Intervals | 52 | 0.939 | 0.893 |
| All features | 78 | 0.980 | 0.964 |

About 83% of the dataset is labeled Romantic. Always predicting Romantic would achieve approximately 83% accuracy, which is why macro-F1 is also reported. These are development-set results, not held-out test results.

### PCA analysis

The first two PCA components explain about **23.6%** of the variance, so a two-dimensional PCA plot cannot be expected to show clear separation for all composers.

| PCA components | Variance explained | Dev accuracy | Dev macro-F1 |
| ---: | ---: | ---: | ---: |
| 2 | 23.6% | 0.204 | 0.234 |
| 30 | 89.4% | 0.551 | 0.479 |
| 78 | 100% | 0.531 | 0.402 |

Performance peaked around 30 components in this development-set experiment. The component count was not evaluated on an independent test set.

### Feature importance

Permutation importance for the RBF SVM was calculated on the development set using macro-F1. Higher-ranked features included vertical-interval bins and pitch-register features such as mean pitch and pitch standard deviation. Importance values varied across repeats, so treat the ranking as exploratory rather than causal evidence about musical style.

## Limitations

- **Small dataset and test set:** 245 pieces overall and 49 test pieces, with as few as three pieces per composer. One prediction changes accuracy by about two percentage points.
- **Class imbalance:** class weights and macro-F1 help address imbalance but do not remove its effects.
- **Piano-roll truncation:** only the first 153.6 seconds are represented, and pitches outside MIDI notes 24–87 are omitted.
- **Limited CNN training data:** the CNN is trained from scratch on only 147 training pieces and underperforms the feature-based models in these experiments.
- **Simplified era labels:** the Classical/Romantic grouping is a project-defined approximation and is strongly imbalanced.
- **Model selection on a small development set:** hyperparameter tuning, model selection, and early stopping use the same 49-piece development set.


## Conclusion

Hand-engineered symbolic features capture useful stylistic information for composer classification. In these experiments, the best reported composer-classification results are around 47–51% accuracy across 11 classes, several times higher than uniform guessing. Interval features appear useful in the exploratory ablation, while the CNN trained on raw piano rolls does not match the feature-based models at this dataset size. The small and imbalanced dataset, small test set, and possibility of related compositions appearing across splits mean the exact scores should be treated as estimates rather than definitive benchmarks.

## Data source

Classical Music MIDI dataset by Soumik Rakshit, Kaggle: <https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi>
