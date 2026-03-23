# Classical ML Pipeline — Sklearn Workout

A beginner-to-production sklearn workflow built as a hands-on exercise to consolidate core scikit-learn concepts:
preprocessing, pipelines, model training, evaluation, and metric reasoning.

---

## 🎯 Project Purpose

This project demonstrates a **complete classical ML workflow** using scikit-learn — from raw data to an evaluated, production-style pipeline.

Built as part of the **BITS Pilani AI & MLOps Engineering Program** learning path, following foundational study of the scikit-learn library.

Each round introduces a new dataset and a new challenge — progressively building from guided practice to full independence.

---

## 🧠 Concepts Covered

| Concept | sklearn Module |
|---|---|
| Train/test splitting (stratified) | `sklearn.model_selection` |
| Feature scaling | `sklearn.preprocessing.StandardScaler` |
| Label encoding | `sklearn.preprocessing.LabelEncoder` |
| Chaining steps into one object | `sklearn.pipeline.Pipeline` |
| Model training | `sklearn.linear_model`, `sklearn.ensemble` |
| Cross-validation | `sklearn.model_selection.cross_val_score` |
| Performance evaluation | `sklearn.metrics` |
| Handling class imbalance | `class_weight='balanced'` |
| Image data as features | `sklearn.datasets.fetch_olivetti_faces` |
| Confusion matrix visualisation | `sklearn.metrics.ConfusionMatrixDisplay` |
| Metric reasoning | Precision vs Recall vs F1 — use-case driven |

---

## 📁 Project Structure

```
├── README.md
├── sklearn_breast_cancer.ipynb       # Round 1 — Guided pipeline, binary classification
├── sklearn_core_workout.ipynb        # Round 2 — Iris, LabelEncoder, feature importance
├── sklearn_digits_workout.ipynb      # Round 3 — Digits, full independence
├── sklearn_winequality_workout.ipynb # Round 4 — Wine Quality, imbalanced classes
└── sklearn_olivetti_faces.ipynb      # Round 5 — Olivetti Faces, face recognition
```

---

## 🏋️ The Five Rounds

### Round 1 — Breast Cancer (Guided)
**Dataset:** `sklearn.datasets.load_breast_cancer`
- Binary classification (malignant vs benign)
- 569 samples, 30 features
- First end-to-end pipeline: split → scale → train → evaluate
- **New concept:** Pipeline, StandardScaler, classification_report

---

### Round 2 — Iris (LabelEncoder + Feature Importance)
**Dataset:** `sklearn.datasets.load_iris`
- Multi-class classification (3 flower species)
- 150 samples, 4 features
- **New concepts:** LabelEncoder for output labels, feature importance from Random Forest
- Key learning: LabelEncoder for target labels; OneHotEncoder for input features

---

### Round 3 — Digits (Full Independence)
**Dataset:** `sklearn.datasets.load_digits`
- Handwritten digit recognition (0–9)
- 1797 samples, 64 features (8×8 pixel images)
- Built entirely without guidance
- **New concept:** Image-as-features (pixel arrays), 10-class classification

---

### Round 4 — Wine Quality (Imbalanced Classes)
**Dataset:** Wine Quality (CSV) — multi-class quality scores
- Real-world imbalanced dataset
- **New concept:** `class_weight='balanced'` to handle skewed class distribution
- Key learning: weighted vs macro F1; when imbalance changes your metric strategy

---

### Round 5 — Olivetti Faces (Face Recognition)
**Dataset:** `sklearn.datasets.fetch_olivetti_faces`
- 400 grayscale face images, 40 people (10 images each)
- 4096 features per image (64×64 pixels, flattened)
- **97.5% accuracy** — model predicts the correct person from a face image
- **New concepts:** High-dimensional image data, face recognition, confusion matrix on 40 classes
- **Metric reasoning:** Precision as lead metric (criminal photo analysis use case); Macro F1 for balanced multi-class

---

## ⚙️ General Workflow (All Rounds)

```
Raw Data
    ↓
Train / Test Split          (stratified, 80/20)
    ↓
StandardScaler              (scale numeric/pixel features)
    ↓
Classifier                  (Logistic Regression or Random Forest)
    ↓
classification_report       (precision, recall, F1 per class)
    ↓
Confusion Matrix            (visualise where errors occur)
    ↓
Metric Reasoning            (identify lead metric for the use case)
```

---

## 🔍 Metric Reasoning Framework

A key skill built across these rounds — choosing the right lead metric is not mechanical, it requires reasoning from the use case:

| Question | Why it matters |
|---|---|
| Is the data balanced or imbalanced? | Determines macro vs weighted F1 |
| What is the cost of a False Positive? | Drives precision vs recall trade-off |
| What is the cost of a False Negative? | Drives recall vs precision trade-off |
| How many classes? | Determines averaging strategy |

**Example (Round 5 — Face Recognition for Criminal Identification):**
- FP = innocent flagged as criminal → highest cost
- Legal principle: better a criminal go free than an innocent be convicted
- **Lead metric: Precision (Macro)** — penalises false accusations directly

---

## 🚀 How to Run

1. Open any notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells in sequence
3. No additional installations required — all libraries available in Colab by default

---

## 📈 Results Summary

| Round | Dataset | Accuracy | Key Challenge |
|---|---|---|---|
| 1 | Breast Cancer | ~96% | First pipeline |
| 2 | Iris | ~97% | Multi-class, feature importance |
| 3 | Digits | ~98% | Full independence |
| 4 | Wine Quality | ~60–65% | Class imbalance |
| 5 | Olivetti Faces | **97.5%** | Image data, 40-class recognition |

---

## 🔑 Key Learnings

> A `Pipeline` chains preprocessing and model training into one deployable object.
> `fit()` on training data. `predict()` on test data. The pipeline handles the rest.

> Scale **after** splitting — never before. Scaling before the split leaks test statistics into training. This is one of the most common sources of data leakage in ML pipelines.

> Simpler models can outperform complex ones on small, clean datasets. Logistic Regression consistently matched or beat Random Forest across these exercises.

> The right metric is decided by the cost of being wrong — not by convention.

---

## 🗺️ What's Next

This exercise is a stepping stone toward:
- Custom preprocessors (`CustomPreprocessor`)
- Multi-column preprocessing with `ColumnTransformer`
- Text classification pipelines with `TfidfVectorizer`
- SciBERT fine-tuning for NLP classification
- Production ML with Ray / Anyscale (MadeWithML)

---

## 👤 Author

Built by Sathya as part of the **BITS Pilani Digital AI Engineering & MLOps Program** learning path.
