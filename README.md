
# 🏦 Home Credit - Credit Risk Model Stability

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)

## 📌 Competition Overview

This Kaggle competition, hosted by **Home Credit**, aims to predict the probability that a client will default on a loan. Unlike traditional competitions focusing solely on AUC or accuracy, this challenge emphasizes **model stability over time**.

In real-world credit scoring, clients’ behaviors evolve—making model stability a key factor. The evaluation metric here combines **AUC (Gini)** with **temporal stability**, rewarding models that maintain consistent predictive performance across **WEEK_NUM** segments.

### 🔍 Evaluation Metric

Final score:

```
Stability Score = mean(gini) + 88.0 * min(0, slope) - 0.5 * std(residuals)
```

Where:
- **gini = 2 * AUC - 1**
- **slope** is the trend of gini scores over time (WEEK_NUM)
- **std(residuals)** captures oscillations in gini over weeks

---

## 📁 Dataset Description

The dataset contains information about credit applicants, derived from both **internal sources** (e.g., applications, bank records) and **external data providers** (e.g., tax registry, credit bureaus).

- `case_id`: Unique identifier for each credit application
- `target`: Whether the client defaulted (1) or not (0)
- `WEEK_NUM`: Weekly grouping of application data for stability evaluation
- `MONTH`: Aggregation support field
- `num_group1` and `num_group2`: Indices for historical records in depth=1 and depth=2 tables
- `date_decision`: Loan decision date

### 📂 Data Depth and Tables

| Depth | Example Tables | Description |
|-------|----------------|-------------|
| 0 | static_0, static_cb_0 | Static per-case info |
| 1 | credit_bureau_a_1, person_1 | Historical data, indexed by `num_group1` |
| 2 | credit_bureau_a_2, applprev_2 | Nested historical data, indexed by `num_group1` and `num_group2` |

---

## 🧮 Feature Definitions

Features in the dataset are derived using encoded transformations:

| Suffix | Transformation |
|--------|----------------|
| P | Days past due (DPD) |
| M | Masked categories |
| A | Amount transformations |
| D | Date-based transformations |
| T / L | Other transformations |

Example: `maxdbddpdtollast6m_4187119P` → Maximum DPD over the last 6 months.

The file `feature_definitions.csv` maps each feature to:
- Source table
- Aggregation logic
- Transformation applied

These definitions are used for consistent and automated feature engineering in the notebook.

---

## 📓 Notebook Overview: `home-credit-lgb-cat-ensemble.ipynb`

This notebook trains an ensemble of **LightGBM** and **CatBoost** models to predict loan defaults while incorporating model **stability**.

### 🔧 Key Features:
- ✅ Aggregation pipelines for depth-1 and depth-2 historical tables
- ✅ Smart feature engineering using `feature_definitions.csv`
- ✅ LightGBM and CatBoost model training with stratified sampling
- ✅ Per-week Gini computation
- ✅ Ensemble blending (averaged predictions)
- ✅ Final submission generation in the required format: `submission.csv`

### 🔍 Stability Strategy

To ensure prediction stability:
- 📅 Weekly-based training-validation splits to monitor drift
- 📉 Per-WEEK_NUM Gini tracking during validation
- 🧪 Penalization-aware hyperparameter tuning
- 📊 Stability visualization to analyze trends and oscillations

> ⚠️ Note: No dataset-specific tuning or hand-crafted feature selection is done — this promotes **generalization** and satisfies the requirements for the **Stability Prize Track**.

---

## 🧠 Requirements

- Python ≥ 3.8
- Pandas, NumPy, Scikit-learn
- LightGBM
- CatBoost
- Matplotlib / Seaborn (for EDA and plots)

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# In Jupyter or Kaggle Notebooks
Open `home-credit-lgb-cat-ensemble.ipynb` and run all cells
```

Outputs:
- `submission.csv` – submission file containing `case_id,score`
- Evaluation charts for weekly Gini scores
