# Machine Learning on Artificially Forged Data

A BTech CSE (AI/ML) Mini-Project evaluating synthetic tabular data generation from minimal seed samples.

---

## 1. Project Overview

Imagine a hospital or a bank that wants to build machine learning models or share data with external researchers. Because of strict privacy regulations (like GDPR and HIPAA) and proprietary constraints, sharing raw customer or patient records is prohibited or heavily restricted. Often, data teams only have access to a very small, carefully anonymized sample (a "seed" dataset).

This project investigates whether modern generative models (like CTGAN and Gaussian Copula) can take a small seed sample (as small as 100 to 2,000 rows) and forge a large, synthetic dataset (5,000 rows) that:
1. Accurately preserves the real data's statistical properties (Fidelity).
2. Successfully trains high-performing downstream machine learning models (Utility).
3. Does not simply memorize or leak genuine customer records (Privacy).

---

## 2. Research Question

> **"How much real data does a generative model need to produce synthetic tabular data that is statistically realistic, useful for training downstream ML models, and safe from copying real records?"**

We test this by evaluating nested seed sizes ($n \in \{100, 250, 500, 1000, 2000\}$) on the Adult Census Income dataset across multiple random seeds, measuring where the quality curves level off.

---

## 3. Pipeline Diagram (ASCII)

```text
+-------------------------------------------------------------------------+
|                              Raw Dataset                                |
|                 Adult Census Income (OpenML ID: 1590)                   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                          Step 1: Load & Clean                           |
|   - Drop exact duplicates BEFORE dropping columns                       |
|   - Drop redundant 'fnlwgt' and 'education' columns                     |
|   - Impute missing values with explicit category "Missing"              |
|   - Assign unique 'row_id' to every record                              |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     Step 3: Stratified 80/20 Split                      |
|                                                                         |
|     +----------------------------+     +--------------------------+     |
|     |  Development Pool (80%)    |     |  Locked Test Set (20%)   |     |
|     |  Used for EDA & Seeds      |     |  *IMMUTABLE / SEALED*    |     |
|     +----------------------------+     |  SHA-256 Manifest Check  |     |
|                   |                    +--------------------------+     |
+-------------------|----------------------------------|------------------+
                    v                                  |
+-----------------------------------------+            |
|       Step 4: Nested Seed Sampling      |            |
|   Sizes: [100, 250, 500, 1000, 2000]    |            |
|   Repeats: Seeds 0, 1, 2                |            |
|   Property: S_100 ⊂ S_250 ⊂ ... ⊂ S_2000|            |
+-----------------------------------------+            |
                    |                                  |
                    v                                  |
+-----------------------------------------+            |
|       Step 5: Generative Engines        |            |
|   - CTGAN (Conditional GAN)             |            |
|   - Gaussian Copula                     |            |
|   - SMOTE-NC (Interpolation Baseline)   |            |
|   -> Synthesizes 5,000 rows per seed    |            |
+-----------------------------------------+            |
                    |                                  |
        +-----------+-----------+                      |
        |                       |                      |
        v                       v                      |
+---------------+       +---------------+              |
|   Step 6:     |       |   Step 7:     |              |
|   Fidelity    |       |   Privacy     |              |
| - KS Test     |       | - DCR vs. Real|              |
| - TVD & Chi^2 |       |   Holdout     |              |
| - Correlation |       | - Exact Match |              |
+---------------+       +---------------+              |
        |                       |                      |
        +-----------+-----------+                      |
                    |                                  |
                    v                                  |
+-----------------------------------------+            |
|         Step 8: Utility Benchmark       |            |
|   Train XGBoost & Random Forest on:     |            |
|   1. TRTR-full  (Real 80% dev pool)     |            |
|   2. TRTR-seed  (Real sparse seed)      |            |
|   3. SMOTE-NC   (Oversampled seed)      |            |
|   4. TSTR-CTGAN (Synthetic CTGAN)       |            |
|   5. TSTR-GC    (Synthetic Copula)      |            |
|                   |                     |            |
|                   +-------------------- v -----------+
|                   Evaluate exclusively on:
|                   LOCKED TEST SET (data/locked/test_real.csv)
|                   Primary Metric: PR-AUC (Average Precision)
+---------------------------------------------------------+
                                     |
                                     v
+---------------------------------------------------------+
|                  Step 9: Criteria Gate                  |
|   Apply criteria.yaml to determine Minimum Useful Seed  |
+---------------------------------------------------------+
                                     |
                                     v
+---------------------------------------------------------+
|                 Step 10: Streamlit App                  |
|   Interactive visualization & custom dataset testing    |
+---------------------------------------------------------+
```

---

## 4. Folder Structure

```text
synthetic-data-project/
├── config.yaml               # Central experiment settings and hyperparameters
├── criteria.yaml             # Success thresholds for fidelity, privacy, and utility
├── requirements.txt          # Python dependencies
├── README.md                 # Complete project documentation and guide
├── PROJECT_BRIEF.md          # Full student project specification
├── .gitignore                # Protects data, models, and virtual environments
├── data/
│   ├── raw/                  # Downloaded raw dataset from OpenML
│   ├── processed/            # Cleaned data and 80% development pool
│   ├── locked/               # 20% locked holdout test set + SHA-256 manifest
│   ├── seeds/                # Nested stratified seed files (n=100 to 2000)
│   └── synthetic/            # Generated synthetic datasets (CTGAN, Copula, SMOTE-NC)
├── src/                      # Clean, simple Python modules (Kathansh & Aryan)
│   └── __init__.py
├── app/                      # Streamlit interactive dashboard
├── tests/                    # Unit and integration test suite
├── results/                  # Metric CSV files (baselines, fidelity, privacy, utility)
├── figures/                  # Publication-ready plots and comparison curves
└── reports/                  # Markdown & HTML reports for viva presentation
```

---

## 5. Setup

### Prerequisites
- Python 3.11 installed.

### Virtual Environment Creation
Run the following in PowerShell (Windows) or Terminal (Linux/macOS):

```bash
# Navigate into the project folder
cd synthetic-data-project

# Create a virtual environment using Python 3.11
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install all required libraries
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. How to Run Each Step

### Step 1: Load and Clean Dataset
- **Module**: `src/clean_data.py` (Track: Kathansh)
- **Command**: `python src/clean_data.py`
- **Tests**: `pytest tests/test_clean_data.py -v`
- **What it does**:
  - Fetches Adult Census Income dataset (OpenML ID: 1590, version 2) and caches it in `data/raw/adult_raw.csv`.
  - Removes 52 exact duplicate records **BEFORE** column removal to preserve genuine data variance (48,842 -> 48,790 rows).
  - Drops redundant columns: `fnlwgt` (census weight) and `education` (duplicate of `education-num`).
  - Imputes all missing values as `"Missing"` category so generative models learn missingness patterns.
  - Inserts a unique `row_id` as the primary key column.
  - Saves cleaned dataset to `data/processed/adult_cleaned.csv`.
- **Status**: Completed & Verified (6/6 tests passing).

### Remaining Pipeline Steps
2. **Step 2 (EDA Report)**: `python src/eda.py`
3. **Step 3 (Locked Split)**: `python src/split_data.py`
4. **Step 4 (Nested Seeds)**: `python src/extract_seeds.py`
5. **Step 5 (Synthesis)**: `python src/generate_synthetic.py`
6. **Step 6 (Fidelity)**: `python src/evaluate_fidelity.py`
7. **Step 7 (Privacy)**: `python src/evaluate_privacy.py`
8. **Step 8 (Downstream ML)**: `python src/evaluate_utility.py`
9. **Step 9 (Decision Gate)**: `python src/analyze_results.py`
10. **Step 10 (Web App)**: `streamlit run app/main.py`

---

## 7. Results

*(Empirical results and comparison tables will be updated upon pipeline execution).*

| Generator / Arm | Min Useful Seed ($n^*$) | PR-AUC vs TRTR-Full | KS Fidelity ($\le 0.10$) | DCR Ratio ($\ge 0.80$) |
| :--- | :--- | :--- | :--- | :--- |
| **CTGAN** | *TBD* | *TBD* | *TBD* | *TBD* |
| **Gaussian Copula** | *TBD* | *TBD* | *TBD* | *TBD* |
| **SMOTE-NC** | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 8. Limitations

1. **Single Dataset Context**: Findings on Adult Census Income might not directly transfer to datasets with extreme high-cardinality categoricals or high-dimensional sparse spaces.
2. **Tabular Only**: Generative architectures evaluated here are specialized for tabular rows, not sequential/time-series or unstructured text.
3. **Computational Trade-offs**: Deep generative networks (CTGAN) require significantly more training iterations and memory than parametric methods like Gaussian Copula.

---

## 9. Future Work

- Expanding to multi-table relational synthesis using HMA (Hierarchical Modeling Algorithm).
- Incorporating Differential Privacy ($\epsilon, \delta$-DP) into the CTGAN discriminator.
- Exploring diffusion-based tabular generative models (e.g., TabDDPM).

---

## 10. Team & Track Attribution

- **Kathansh** (*Data Analysis Track*):
  - Data loading, cleaning, schema validation, and missing value encoding.
  - Development pool EDA and feature distribution analysis.
  - Nested stratified seed extraction engine ($S_{100} \subset \dots \subset S_{2000}$).
  - Statistical fidelity auditing (KS tests, TVD, Chi-square, correlation drift).
  - Privacy audits (Distance to Closest Record and exact copy detection).

- **Aryan** (*GenAI & ML Track*):
  - Generative synthesis pipeline (CTGAN, Gaussian Copula, SMOTE-NC).
  - Downstream classification modeling (XGBoost, Random Forest).
  - 5-Arm benchmarking harness (TRTR vs. TSTR).
  - Criteria rule enforcement and minimum useful seed discovery.
  - Interactive Streamlit dashboard deployment.

---

## 11. References

1. Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling Tabular data using Conditional GAN*. NeurIPS.
2. Patki, N., Wedge, R., & Veeramachaneni, K. (2016). *The Synthetic Data Vault*. IEEE International Conference on Data Science and Advanced Analytics (DSAA).
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
4. Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid*. KDD (Adult Census dataset).
