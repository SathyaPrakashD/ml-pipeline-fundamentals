# ml-pipeline-fundamentals

A beginner-to-production sklearn workflow built as a hands-on exercise to consolidate core scikit-learn concepts: preprocessing, pipelines, model training, evaluation, and metric reasoning.

---

## 🎯 Project Purpose

This project demonstrates a complete classical ML workflow using scikit-learn — from raw data to an evaluated, production-style pipeline. Built as part of the **BITS Pilani AI & MLOps Engineering Program** learning path, following foundational study of the scikit-learn library.

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
| Text classification | `sklearn.feature_extraction.text.TfidfVectorizer` |
| Regression metrics | MAE, RMSE, R² — continuous target prediction |
| Multi-type preprocessing | `sklearn.compose.ColumnTransformer` |
| Custom transformers | `BaseEstimator`, `TransformerMixin` |
| Missing value imputation | `sklearn.impute.SimpleImputer` |
| Regularisation | Ridge, Lasso regression |
| Feature engineering | Derived features, multicollinearity removal |
| Pre-training data validation | Schema, semantic, null, distribution checks |
| Missingness indicator pattern | Impute + flag for missing value signals |
| Data leakage prevention | Scaler fit on training data only |
| Evidence-based model comparison | Baseline → Model A → Model B progression |
| Data lineage tracking | Explicit data lineage notes in pipelines |

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── 01_breast_cancer_pipeline.ipynb          # Round 1  – Guided pipeline, binary classification
├── 02_iris_classification.ipynb             # Round 2  – Iris, full independence
├── 03_sklearn_core_workout.ipynb            # Round 3  – Core sklearn workout, feature importance
├── 04_digits_recognition.ipynb             # Round 4  – Digits, image-as-features, 10-class
├── 05_wine_quality_imbalanced.ipynb         # Round 5  – Wine Quality, imbalanced classes
├── 06_olivetti_faces_recognition.ipynb      # Round 6  – Olivetti Faces, face recognition
├── 07_fetch_20newsgroups.ipynb              # Round 7  – 20 Newsgroups, text classification
├── 08_fetch_california_housing.ipynb        # Round 8  – California Housing, regression
├── 09_ames_housing_regression.ipynb         # Round 9  – Ames Housing, ColumnTransformer + custom transformer
├── 10_titanic_classification.ipynb          # Round 10 – Titanic, mixed data, ColumnTransformer
├── 11_diabetes_progression_regression.ipynb # Round 11 – Diabetes, regularisation + feature engineering
├── 12_heart_disease_classification.ipynb    # Round 12 – Heart Disease, binary classification
├── 13_insurance_charges_regression.ipynb    # Round 13 – Insurance Charges, regression
└── 14_heart_disease_data_engineering.ipynb  # Round 14 – Heart Disease, data engineering overlay
```

---

## 🏋️ The Fourteen Rounds

### Round 1 — Breast Cancer (Guided)

- **Dataset:** `sklearn.datasets.load_breast_cancer`
- - Binary classification (malignant vs benign)
  - - 569 samples, 30 features
    - - First end-to-end pipeline: split → scale → train → evaluate
      - - **New concepts:** `Pipeline`, `StandardScaler`, `classification_report`
       
        - ---

        ### Round 2 — Iris (Full Independence)

        - **Dataset:** `sklearn.datasets.load_iris`
        - - Multi-class classification (3 flower species)
          - - 150 samples, 4 features
            - - Built independently end-to-end
             
              - ---

              ### Round 3 — sklearn Core Workout (LabelEncoder + Feature Importance)

              - **Dataset:** `sklearn.datasets.load_wine`
              - - **New concepts:** `LabelEncoder` for output labels, feature importance from Random Forest
                - - **Key learning:** `LabelEncoder` for target labels; `OneHotEncoder` for input features
                 
                  - ---

                  ### Round 4 — Digits (Full Independence)

                  - **Dataset:** `sklearn.datasets.load_digits`
                  - - Handwritten digit recognition (0–9)
                    - - 1,797 samples, 64 features (8×8 pixel images)
                      - - Built entirely without guidance
                        - - **New concept:** Image-as-features (pixel arrays), 10-class classification
                         
                          - ---

                          ### Round 5 — Wine Quality (Imbalanced Classes)

                          - **Dataset:** Wine Quality (CSV) — multi-class quality scores
                          - - Real-world imbalanced dataset
                            - - **New concept:** `class_weight='balanced'` to handle skewed class distribution
                              - - **Key learning:** weighted vs macro F1; when imbalance changes your metric strategy
                               
                                - ---

                                ### Round 6 — Olivetti Faces (Face Recognition)

                                - **Dataset:** `sklearn.datasets.fetch_olivetti_faces`
                                - - 400 grayscale face images, 40 people (10 images each)
                                  - - 4,096 features per image (64×64 pixels, flattened)
                                    - - 97.5% accuracy — model predicts the correct person from a face image
                                      - - **New concepts:** High-dimensional image data, face recognition, confusion matrix on 40 classes
                                        - - **Metric reasoning:** Precision as lead metric (criminal photo analysis use case); Macro F1 for balanced evaluation
                                         
                                          - ---

                                          ### Round 7 — 20 Newsgroups (Text Classification)

                                          - **Dataset:** `sklearn.datasets.fetch_20newsgroups`
                                          - - 18,846 newsgroup posts across 20 topic categories
                                            - - Text classification: predict the newsgroup category from raw document text
                                              - - **New concepts:** TF-IDF vectorisation (`TfidfVectorizer`), sparse matrix features, text-as-features pipeline
                                                - - Models compared: Logistic Regression vs Random Forest
                                                  - - ~72% accuracy (macro F1) — Logistic Regression wins on sparse text data
                                                    - - **Key learning:** text pipelines require a vectoriser step before any scaling; sparse matrices behave differently from dense arrays
                                                     
                                                      - ---

                                                      ### Round 8 — California Housing (Regression)

                                                      - **Dataset:** `sklearn.datasets.fetch_california_housing`
                                                      - - 20,000 California districts, 8 features (median income, house age, rooms, etc.)
                                                        - - First regression task — predicting continuous house prices
                                                          - - **New concepts:** Regression metrics (MAE, RMSE, R²), `LinearRegression`, `RandomForestRegressor`
                                                            - - Feature correlation analysis and multicollinearity exploration (heatmap)
                                                              - - Results — Linear Regression: R²=0.58, RMSE=0.75 → Random Forest: R²=0.82, RMSE=0.50 (42% more variance explained)
                                                                - - **Key learning:** R² measures explained variance; RMSE penalises large errors; tree-based models handle non-linear relationships naturally
                                                                 
                                                                  - ---

                                                                  ### Round 9 — Ames Housing (ColumnTransformer + Custom Transformer)

                                                                  - **Dataset:** Ames Housing (CSV) — 1,460 houses, 81 columns
                                                                  - - Real-world regression with mixed data (43 categorical + 38 numeric columns)
                                                                    - - Heavy missing values requiring imputation strategy
                                                                      - - **New concepts:** `ColumnTransformer` for mixed-type pipelines, custom `TotalAreaTransformer` (`BaseEstimator` + `TransformerMixin`), feature engineering
                                                                        - - Results — Baseline Random Forest: MAE=$17,526, R²=0.89
                                                                          - - **Key learning:** `ColumnTransformer` becomes a necessity (not theory) with mixed-type real data; custom transformers integrate seamlessly into sklearn pipelines
                                                                           
                                                                            - ---

                                                                            ### Round 10 — Titanic (Mixed Data + ColumnTransformer Classification)

                                                                            - **Dataset:** Titanic passenger manifest — 891 passengers, 8 features after cleaning
                                                                            - - Binary classification: predict survival (1 = survived, 0 = did not survive)
                                                                              - - Mixed data: numeric (Age, Fare, Pclass) + categorical (Sex, Embarked)
                                                                                - - **New concepts:** `ColumnTransformer` for classification, `OneHotEncoder`, missing value imputation (Age, Embarked)
                                                                                  - - Mild class imbalance (62% did not survive, 38% survived) — handled via `class_weight='balanced'`
                                                                                    - - Results — Accuracy: 82%, Recall (survived): 0.76, Precision: 0.80, F1: 0.78
                                                                                      - - **Key learning:** Sex is the strongest survival predictor (74% female vs 19% male survival rate); Recall chosen as lead metric to minimise missed survivors
                                                                                       
                                                                                        - ---

                                                                                        ### Round 11 — Diabetes Progression (Regularisation + Feature Engineering)

                                                                                        - **Dataset:** `sklearn.datasets.load_diabetes` — 442 patients, 10 features
                                                                                        - - Regression: predict continuous disease progression score (higher = worse)
                                                                                          - - Clean dataset — no missing values, all numeric features
                                                                                            - - **New concepts:** Ridge regression, Lasso, regularisation as a tool; feature engineering; multicollinearity removal
                                                                                              - - Optimisation journey: Baseline LR (R²=0.45) → +Feature Engineering (0.48) → +Drop multicollinear (0.49) → +Ridge/Lasso (marginal)
                                                                                                - - Final model: MAE=42, RMSE=52, R²=0.49 — 49% of progression variance explained
                                                                                                  - - **Key learning:** the remaining 51% variance is driven by factors outside the dataset (diet, genetics, lifestyle)
                                                                                                   
                                                                                                    - ---
                                                                                                    
                                                                                                    ### Round 12 — Heart Disease Classification (OpenML + fetch_openml)
                                                                                                    
                                                                                                    - **Dataset:** UCI Heart Disease (Heart Statlog) via OpenML — 270 patients, 13 features
                                                                                                    - - Binary classification: predict presence or absence of heart disease
                                                                                                      - - Clinical features: age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more
                                                                                                        - - **New concepts:** `fetch_openml` for dataset loading, `DummyClassifier` baseline, model comparison (Logistic Regression vs Random Forest)
                                                                                                          - - Models used: DummyClassifier → RandomForestClassifier → LogisticRegression
                                                                                                            - - **Key learning:** always establish a `DummyClassifier` baseline before comparing real models; feature importance from clinical data reveals medical risk factors
                                                                                                             
                                                                                                              - ---
                                                                                                              
                                                                                                              ### Round 13 — Insurance Charges (Regression with Mixed Features)
                                                                                                              
                                                                                                              - **Dataset:** insurance.csv — 1,338 records, 7 features (age, sex, BMI, children, smoker, region, charges)
                                                                                                              - - Regression: predict annual medical insurance charges from patient demographic and lifestyle data
                                                                                                                - - Mixed data: numeric (age, BMI, children) + categorical (sex, smoker, region)
                                                                                                                  - - **New concepts:** real-world CSV regression, smoking status as dominant predictor, feature importance in insurance domain
                                                                                                                    - - Models used: DummyRegressor → LinearRegression → RandomForestRegressor
                                                                                                                      - - **Key learning:** a single binary feature (smoker) can dominate all other predictors; categorical encoding is essential for regression with mixed data
                                                                                                                       
                                                                                                                        - ---
                                                                                                                        
                                                                                                                        ### Round 14 — Heart Disease Data Engineering Overlay
                                                                                                                        
                                                                                                                        - **Dataset:** UCI Heart Disease (Cleveland) — same dataset as Round 12
                                                                                                                        - - **Purpose:** This notebook is not about modelling — it is about data engineering discipline
                                                                                                                          - - **New concepts:** pre-training validation block (schema, semantic, null, distribution checks); systematic preprocessing via `ColumnTransformer` + `Pipeline`; leakage prevention (scaler fit on training data only); missingness indicator pattern (impute + flag); evidence-based model comparison (Baseline → Model A → Model B); explicit data lineage notes
                                                                                                                            - - **Week 4 reference:** BITS Pilani AI Engineering — Module 2, Week 4
                                                                                                                              - - **Key learning:** production ML requires rigorous data validation and leakage prevention before any model is trained; data engineering is a discipline separate from modelling
                                                                                                                               
                                                                                                                                - ---
                                                                                                                                
                                                                                                                                ## ⚙️ General Workflow (All Rounds)
                                                                                                                                
                                                                                                                                ```
                                                                                                                                Raw Data
                                                                                                                                  ↓
                                                                                                                                Train / Test Split (stratified, 80/20)
                                                                                                                                  ↓
                                                                                                                                Preprocessing (StandardScaler / ColumnTransformer — fit on train only)
                                                                                                                                  ↓
                                                                                                                                Model Training (within Pipeline)
                                                                                                                                  ↓
                                                                                                                                Evaluation (accuracy / F1 / precision / recall / R² / RMSE / MAE)
                                                                                                                                  ↓
                                                                                                                                Metric Reasoning (choose lead metric based on use case)
                                                                                                                                ```
                                                                                                                                
                                                                                                                                ---
                                                                                                                                
                                                                                                                                ## 🔍 Metric Reasoning Framework
                                                                                                                                
                                                                                                                                A key skill built across these rounds — choosing the right lead metric is not mechanical; it requires reasoning about the cost of being wrong.
                                                                                                                                
                                                                                                                                | Question | Why it matters |
                                                                                                                                |---|---|
                                                                                                                                | Is the data balanced or imbalanced? | Determines macro vs weighted F1 |
                                                                                                                                | What is the cost of a False Positive? | Drives precision vs recall trade-off |
                                                                                                                                | What is the cost of a False Negative? | Drives recall vs precision trade-off |
                                                                                                                                | How many classes? | Determines averaging strategy |
                                                                                                                                | Is the target continuous? | Use MAE / RMSE / R² instead of accuracy |
                                                                                                                                
                                                                                                                                **Example (Round 6 — Face Recognition for Criminal Identification):**
                                                                                                                                - FP = innocent person flagged as criminal → highest cost
                                                                                                                                - - Legal principle: better a criminal go free than an innocent be convicted
                                                                                                                                  - - **Lead metric:** Precision (Macro) — penalises false accusations directly
                                                                                                                                   
                                                                                                                                    - **Example (Round 10 — Titanic Survival Prediction):**
                                                                                                                                    - - FN = survivor classified as non-survivor → consequences are severe
                                                                                                                                      - - **Lead metric:** Recall — minimise missed survivors
                                                                                                                                       
                                                                                                                                        - **Example (Round 12 — Heart Disease Classification):**
                                                                                                                                        - - FN = patient with heart disease classified as healthy → potentially fatal miss
                                                                                                                                          - - **Lead metric:** Recall — catching every at-risk patient is the priority
                                                                                                                                           
                                                                                                                                            - **Example (Rounds 8, 11 & 13 — Housing / Disease / Insurance Prediction):**
                                                                                                                                            - - No false positives/negatives — target is continuous
                                                                                                                                              - - R² explains how much variance the model captures
                                                                                                                                                - - RMSE penalises large prediction errors more than MAE
                                                                                                                                                  - - **Lead metric:** RMSE for model quality; MAE for practical error magnitude
                                                                                                                                                   
                                                                                                                                                    - ---
                                                                                                                                                    
                                                                                                                                                    ## 🚀 How to Run
                                                                                                                                                    
                                                                                                                                                    **Option 1 — Google Colab (recommended):**
                                                                                                                                                    Click any "Open in Colab" badge in the individual notebooks above.
                                                                                                                                                    
                                                                                                                                                    **Option 2 — Run locally:**
                                                                                                                                                    ```bash
                                                                                                                                                    pip install -r requirements.txt
                                                                                                                                                    jupyter notebook
                                                                                                                                                    ```
                                                                                                                                                    
                                                                                                                                                    ---
                                                                                                                                                    
                                                                                                                                                    ## 📈 Results Summary
                                                                                                                                                    
                                                                                                                                                    | Round | Dataset | Metric | Score | Key Challenge |
                                                                                                                                                    |---|---|---|---|---|
                                                                                                                                                    | 1 | Breast Cancer | Accuracy | ~96% | First pipeline |
                                                                                                                                                    | 2 | Iris | Accuracy | ~97% | Full independence |
                                                                                                                                                    | 3 | Wine (sklearn) | Accuracy | ~97% | Feature importance |
                                                                                                                                                    | 4 | Digits | Accuracy | ~98% | Image-as-features |
                                                                                                                                                    | 5 | Wine Quality | Weighted F1 | ~60–65% | Class imbalance |
                                                                                                                                                    | 6 | Olivetti Faces | Accuracy | 97.5% | Image data, 40-class recognition |
                                                                                                                                                    | 7 | 20 Newsgroups | Macro F1 | ~72% | Text features, 20-class classification |
                                                                                                                                                    | 8 | California Housing | R² / RMSE | 0.82 / 0.50 | First regression task |
                                                                                                                                                    | 9 | Ames Housing | R² / MAE | 0.89 / $17,526 | Mixed data, custom transformer |
                                                                                                                                                    | 10 | Titanic | Accuracy / F1 | 82% / 0.78 | Mixed data, ColumnTransformer classification |
                                                                                                                                                    | 11 | Diabetes Progression | R² / RMSE | 0.49 / 52 | Regularisation, feature engineering |
                                                                                                                                                    | 12 | Heart Disease | Accuracy / Recall | See notebook | Binary classification, OpenML, DummyClassifier baseline |
                                                                                                                                                    | 13 | Insurance Charges | R² / RMSE | See notebook | Mixed-feature regression, smoker dominance |
                                                                                                                                                    | 14 | Heart Disease (DE) | — | — | Data engineering discipline, leakage prevention |
                                                                                                                                                    
                                                                                                                                                    ---
                                                                                                                                                    
                                                                                                                                                    ## 🔑 Key Learnings
                                                                                                                                                    
                                                                                                                                                    - A `Pipeline` chains preprocessing and model training into one deployable object. `fit()` on training data. `predict()` on test data.
                                                                                                                                                    - - **Scale after splitting — never before.** Scaling before the split leaks test statistics into training and invalidates evaluation.
                                                                                                                                                      - - Simpler models can outperform complex ones on small, clean datasets. Logistic Regression consistently beats Random Forest on text and small tabular data.
                                                                                                                                                        - - Text data requires a vectoriser (`TfidfVectorizer`) as the first pipeline step — not a scaler. Sparse matrices are very different from dense feature arrays.
                                                                                                                                                          - - Real-world datasets have mixed types. `ColumnTransformer` is the standard tool for applying different preprocessing steps to numeric and categorical columns simultaneously.
                                                                                                                                                            - - Custom transformers (`BaseEstimator` + `TransformerMixin`) plug seamlessly into sklearn pipelines for feature engineering, making bespoke transformations reproducible and deployment-ready.
                                                                                                                                                              - - Regularisation (Ridge, Lasso) adds marginal gains once manual multicollinearity removal is done — but it is still a critical tool for preventing overfitting on unseen data.
                                                                                                                                                                - - Always establish a `DummyClassifier` or `DummyRegressor` baseline before comparing real models. A model that cannot beat a naive baseline is not learning anything useful.
                                                                                                                                                                  - - Data engineering is a discipline separate from modelling. Pre-training validation, leakage prevention, and data lineage tracking are production requirements — not optional extras.
                                                                                                                                                                    - - The right metric is decided by the **cost of being wrong** — not by convention. Classification uses accuracy/F1/precision/recall depending on the use case. Regression uses MAE/RMSE/R².
                                                                                                                                                                     
                                                                                                                                                                      - ---
                                                                                                                                                                      
                                                                                                                                                                      ## 🗺️ What's Next
                                                                                                                                                                      
                                                                                                                                                                      This exercise is a stepping stone toward:
                                                                                                                                                                      
                                                                                                                                                                      - Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
                                                                                                                                                                      - - Gradient Boosting models (XGBoost, LightGBM)
                                                                                                                                                                        - - Text classification pipelines with `TfidfVectorizer` into deep learning
                                                                                                                                                                          - - SciBERT fine-tuning for NLP classification
                                                                                                                                                                            - - Production ML with Ray / Anyscale
                                                                                                                                                                             
                                                                                                                                                                              - ---
                                                                                                                                                                              
                                                                                                                                                                              ## 👤 Author
                                                                                                                                                                              
                                                                                                                                                                              **Sathya Prakash** — [LinkedIn](https://www.linkedin.com/in/sathyaprakashd/)
                                                                                                                                                                              
                                                                                                                                                                              Built as part of the **BITS Pilani Digital AI Engineering & MLOps Program** learning path.
                                                                                                                                                                              
                                                                                                                                                                              ---
                                                                                                                                                                              
                                                                                                                                                                              ## 📄 License
                                                                                                                                                                              
                                                                                                                                                                              This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
