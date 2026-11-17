# PROJECT BRIEF: Machine Learning on Artificially Forged Data

**Student Context**: 3rd-year BTech AI/ML mini-project. Designed to be explained line-by-line during a viva examination with simple, beginner-readable code.

---

## General Rules (To follow in every step of this project)
- **Python Version**: Python 3.11
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `sdv`, `scipy`, `matplotlib`, `seaborn`, `streamlit`, `pyyaml`, `pytest`.
- **Code Style**: Simple functions, no complex class hierarchies, no clever tricks. Clear and descriptive variable names.
- **Documentation**: Every `.py` file starts with a docstring explaining in plain English what the file does and why it exists. Comments explain **WHY**, not just what.
- **Configuration**: All settings come from `config.yaml` (no hard-coded numbers in code).
- **GOLDEN RULE**: The locked test set is used **ONLY** for the final model evaluation. It must never be used for cleaning decisions, fitting encoders, seed selection, training generators, or hyperparameter tuning. Always verify its SHA-256 hash before loading it.
- **Reproducibility**: Fix random seeds everywhere (`numpy`, `random`, `torch`, `sklearn`, `xgboost`).
- **Workflow Protocol**: After every step: run the code, show the output, update `README.md`, and explain what was built in 5–10 simple sentences. Do not start the next step until explicitly asked.

---

## Project Summary
- **Research Question**: How much real data does a generator need to produce synthetic tabular data that is realistic, useful for training ML models, and does not copy real records?
- **Dataset**: Adult Census Income (OpenML `"adult"`, version 2), binary target `income >50K` (~24% positive class).

---

## End-to-End Pipeline
1. **Load + Clean Data**:
   - Drop `fnlwgt` and `education` (education-num is retained as numerical).
   - Missing values become the categorical value `"Missing"`.
   - Drop exact duplicate rows **BEFORE** dropping columns.
   - Add a `row_id` column for tracking.
2. **EDA Report**: Generated on the development pool only.
3. **Stratified Split**:
   - 80% Development pool.
   - 20% Locked test set + SHA-256 manifest.
4. **Nested Stratified Seeds**:
   - Seed sizes: 100, 250, 500, 1000, 2000 drawn from the development pool.
   - Strict subset nesting (each smaller seed is strictly contained within the larger one).
   - Repeated across random seeds 0, 1, 2.
5. **Generators on Each Seed**:
   - CTGAN (`sdv.single_table.CTGANSynthesizer`) with fixed total training steps.
   - Gaussian Copula (`sdv.single_table.GaussianCopulaSynthesizer`).
   - SMOTE-NC (`imblearn.over_sampling.SMOTENC`).
   - Each generator produces 5,000 synthetic rows.
6. **Fidelity Evaluation**:
   - Numerical features: Two-sample Kolmogorov-Smirnov (KS) test.
   - Categorical features: Total Variation Distance (TVD) and Chi-square test.
   - Multivariate: Correlation matrix difference.
   - Category coverage (synthetic vs. dev pool).
7. **Privacy Evaluation**:
   - Distance to Closest Record (DCR) compared against a real-holdout baseline.
   - Exact-match rate check.
8. **Utility Evaluation**:
   - Classifiers: Random Forest and XGBoost.
   - 5 Experimental Arms:
     1. `TRTR-full` (Train Real Full, Test Real Locked)
     2. `TRTR-seed` (Train Real Seed, Test Real Locked)
     3. `SMOTE-NC` (Train SMOTE-NC, Test Real Locked)
     4. `TSTR-CTGAN` (Train Synthetic CTGAN, Test Real Locked)
     5. `TSTR-GC` (Train Synthetic Gaussian Copula, Test Real Locked)
   - Evaluated on locked test set. Primary metric: PR-AUC (`average_precision_score`); secondary metrics: ROC-AUC, F1, Precision, Recall, Accuracy.
9. **Analysis & Criteria**:
   - Mean ± standard deviation over random seeds.
   - Plots vs. seed size.
   - Apply `criteria.yaml` to determine the minimum useful seed size.
10. **Streamlit App**:
    - Pages: About, Experiment Results, Try It Yourself (interactive CSV upload).

---

## Team Roles
- **Kathansh** (Data Analysis Track): Data loading, cleaning, EDA, seed extraction, statistical fidelity audits, privacy/DCR audits.
- **Aryan** (GenAI & ML Track): Generative synthesis (CTGAN, Gaussian Copula, SMOTE-NC), ML classification models, benchmarking harness, Streamlit application.
