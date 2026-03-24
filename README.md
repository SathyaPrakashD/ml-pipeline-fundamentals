# ml-pipeline-fundamentals

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

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
├── requirements.txt
├── LICENSE
├── .gitignore
├── 01_breast_cancer_pipeline.ipynb     # Round 1 — Guided pipeline, binary classification
├── 02_iris_classification.ipynb        # Round 2 — Iris, full independence
├── 03_sklearn_core_workout.ipynb       # Round 3 — Core sklearn workout, feature importance
├── 04_digits_recognition.ipynb         # Round 4 — Digits, image-as-features, 10-class
├── 05_wine_quality_imbalanced.ipynb    # Round 5 — Wine Quality, imbalanced classes
└── 06_olivetti_faces_recognition.ipynb # Round 6 — Olivetti Faces, face recognition
```

---

## 🏋️ The Six Rounds

### Round 1 — Breast Cancer (Guided)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/01_breast_cancer_pipeline.ipynb)

**Dataset:** `sklearn.datasets.load_breast_cancer`
- Binary classification (malignant vs benign)
- - 569 samples, 30 features
  - - First end-to-end pipeline: split → scale → train → evaluate
    - - **New concept:** Pipeline, StandardScaler, classification_report
     
      - ---

      ### Round 2 — Iris (Full Independence)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/02_iris_classification.ipynb)

      **Dataset:** `sklearn.datasets.load_iris`
      - Multi-class classification (3 flower species)
      - - 150 samples, 4 features
        - - Built independently end-to-end
         
          - ---

          ### Round 3 — sklearn Core Workout (LabelEncoder + Feature Importance)
          [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/03_sklearn_core_workout.ipynb)

          **Dataset:** `sklearn.datasets.load_wine`
          - **New concepts:** LabelEncoder for output labels, feature importance from Random Forest
          - - Key learning: LabelEncoder for target labels; OneHotEncoder for input features
           
            - ---

            ### Round 4 — Digits (Full Independence)
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/04_digits_recognition.ipynb)

            **Dataset:** `sklearn.datasets.load_digits`
            - Handwritten digit recognition (0–9)
            - - 1797 samples, 64 features (8×8 pixel images)
              - - Built entirely without guidance
                - - **New concept:** Image-as-features (pixel arrays), 10-class classification
                 
                  - ---

                  ### Round 5 — Wine Quality (Imbalanced Classes)
                  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/05_wine_quality_imbalanced.ipynb)

                  **Dataset:** Wine Quality (CSV) — multi-class quality scores
                  - Real-world imbalanced dataset
                  - - **New concept:** `class_weight='balanced'` to handle skewed class distribution
                    - - Key learning: weighted vs macro F1; when imbalance changes your metric strategy
                     
                      - ---

                      ### Round 6 — Olivetti Faces (Face Recognition)
                      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathyaPrakashD/ml-pipeline-fundamentals/blob/main/06_olivetti_faces_recognition.ipynb)

                      **Dataset:** `sklearn.datasets.fetch_olivetti_faces`
                      - 400 grayscale face images, 40 people (10 images each)
                      - - 4096 features per image (64×64 pixels, flattened)
                        - - 97.5% accuracy — model predicts the correct person from a face image
                          - - **New concepts:** High-dimensional image data, face recognition, confusion matrix on 40 classes
                            - - Metric reasoning: Precision as lead metric (criminal photo analysis use case); Macro F1 for balanced multi-class
                             
                              - ---

                              ## ⚙️ General Workflow (All Rounds)

                              ```
                              Raw Data
                              ↓
                              Train / Test Split (stratified, 80/20)
                              ↓
                              StandardScaler (scale numeric/pixel features)
                              ↓
                              Classifier (Logistic Regression or Random Forest)
                              ↓
                              classification_report (precision, recall, F1 per class)
                              ↓
                              Confusion Matrix (visualise where errors occur)
                              ↓
                              Metric Reasoning (identify lead metric for the use case)
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

                              **Example (Round 6 — Face Recognition for Criminal Identification):**
                              - FP = innocent flagged as criminal → highest cost
                              - - Legal principle: better a criminal go free than an innocent be convicted
                                - - Lead metric: **Precision (Macro)** — penalises false accusations directly
                                 
                                  - ---

                                  ## 🚀 How to Run

                                  **Option 1 — Google Colab (recommended):** Click any "Open in Colab" badge above.

                                  **Option 2 — Run locally:**
                                  ```bash
                                  pip install -r requirements.txt
                                  jupyter notebook
                                  ```

                                  ---

                                  ## 📈 Results Summary

                                  | Round | Dataset | Accuracy | Key Challenge |
                                  |---|---|---|---|
                                  | 1 | Breast Cancer | ~96% | First pipeline |
                                  | 2 | Iris | ~97% | Full independence |
                                  | 3 | Wine (sklearn) | ~97% | Feature importance |
                                  | 4 | Digits | ~98% | Image-as-features |
                                  | 5 | Wine Quality | ~60–65% | Class imbalance |
                                  | 6 | Olivetti Faces | 97.5% | Image data, 40-class recognition |

                                  ---

                                  ## 🔑 Key Learnings

                                  > A `Pipeline` chains preprocessing and model training into one deployable object. `fit()` on training data. `predict()` on test data. The pipeline handles the rest.
                                  >
                                  > > Scale **after** splitting — never before. Scaling before the split leaks test statistics into training. This is one of the most common sources of data leakage in ML pipelines.
                                  > >
                                  > > > Simpler models can outperform complex ones on small, clean datasets. Logistic Regression consistently matched or beat Random Forest across these exercises.
                                  > > >
                                  > > > > The right metric is decided by the **cost of being wrong** — not by convention.
                                  > > > >
                                  > > > > ---
                                  > > > >
                                  > > > > ## 🗺️ What's Next
                                  > > > >
                                  > > > > This exercise is a stepping stone toward:
                                  > > > > - Custom preprocessors (`CustomPreprocessor`)
                                  > > > > - - Multi-column preprocessing with `ColumnTransformer`
                                  > > > >   - - Text classification pipelines with `TfidfVectorizer`
                                  > > > >     - - SciBERT fine-tuning for NLP classification
                                  > > > >       - - Production ML with Ray / Anyscale
                                  > > > >        
                                  > > > >         - ---
                                  > > > >
                                  > > > > ## 👤 Author
                                  > > > >
                                  > > > > **Sathya Prakash** · [LinkedIn](https://www.linkedin.com/in/sathyaprakashd)
                                  > > > >
                                  > > > > Built as part of the **BITS Pilani Digital AI Engineering & MLOps Program** learning path.
                                  > > > >
                                  > > > > ---
                                  > > > >
                                  > > > > ## 📄 License
                                  > > > >
                                  > > > > This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
