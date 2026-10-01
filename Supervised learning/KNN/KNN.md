
# 🏦 Loan Default & Approval Prediction using KNN and Naïve Bayes

A comparative Machine Learning classification project designed to assess applicant credit risk and predict loan status (`Approved` vs. `Rejected`) using **K-Nearest Neighbors (KNN)** and **Gaussian Naïve Bayes (GNB)**[cite: 63, 64].

---

## 📌 Project Overview

When evaluating personal loan applications, financial institutions must balance risk minimization (avoiding defaults) with revenue optimization (approving qualified borrowers)[cite: 63, 64]. 

This project explores an end-to-end classification pipeline:

```

Data Ingestion ➔ Data Cleaning ➔ EDA ➔ One-Hot Encoding ➔ Train/Test Split
➔ Class Balancing (SMOTE) ➔ Feature Scaling ➔ Modeling (GNB & KNN) ➔ Evaluation

```
[cite: 63, 64]

---

## 📊 Dataset Description

The dataset (`loan_data (1).csv`) consists of **1,000 applicant records** across **12 attributes**[cite: 63, 64]:

| Feature | Data Type | Description |
| :--- | :--- | :--- |
| `applicant_id` | Identifier | Unique applicant ID (dropped during modeling to avoid high cardinality)[cite: 63, 64] |
| `age` | Numeric | Age of the applicant in years[cite: 63, 64] |
| `annual_income` | Numeric | Annual income in INR[cite: 63, 64] |
| `loan_amount` | Numeric | Total principal loan amount requested[cite: 63, 64] |
| `monthly_emi` | Numeric | Expected monthly installment[cite: 63, 64] |
| `credit_score` | Numeric | CIBIL/Credit bureau score[cite: 63, 64] |
| `existing_loans` | Numeric | Number of currently active loans[cite: 63, 64] |
| `city` | Categorical | Metro city (`Bangalore`, `Delhi`, `Mumbai`, `Kolkata`, `Pune`, etc.)[cite: 63, 64] |
| `education` | Categorical | Education level (`Graduate`, `Not Graduate`)[cite: 63, 64] |
| `employment_type` | Categorical | Employment category (`Salaried`, `Self-Employed`, `Business Owner`, `Freelancer`)[cite: 63, 64] |
| `has_property` | Binary | Asset ownership indicator (`0` = No, `1` = Yes)[cite: 63, 64] |
| **`loan_status`** | **Binary Target** | Target label: `Approved` (Encoded as `0`), `Rejected` (Encoded as `1`)[cite: 63, 64] |

---

## 🛠️ Data Preprocessing Pipeline

1. **Handling Missing Values:**
   - Identified missing entries in `monthly_emi` (20 rows) and `credit_score` (30 rows)[cite: 63].
   - Listwise removal (`df.dropna(inplace=True)`) produced a cleaned dataset of **950 complete records**[cite: 63, 64].

2. **Feature Encoding & ID Removal:**
   - Dropped `applicant_id` to prevent dimensionality explosion[cite: 64].
   - Converted categorical variables (`city`, `education`, `employment_type`) into dummy numerical columns using `pd.get_dummies(..., drop_first=True)`[cite: 64].
   - Encoded the target `loan_status` using `LabelEncoder` (`Approved: 0`, `Rejected: 1`)[cite: 64].

3. **Train-Test Partition:**
   - Split dataset with `test_size=0.33` and `random_state=42`[cite: 64].
   - Train set: 636 samples | Test set: 314 samples[cite: 64].

4. **Class Imbalance Handling (SMOTE):**
   - The dataset has an inherent class imbalance (~81% Approved vs. ~19% Rejected)[cite: 63, 64].
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training split to synthesize minority samples, resulting in **1,048 balanced training instances** (524 per class)[cite: 64].

5. **Feature Standardization:**
   - Applied `StandardScaler` to ensure scale-sensitive distance calculations in KNN are not dominated by large-scale numeric fields like `annual_income` and `loan_amount`[cite: 64].

---

## 🤖 Model Comparison & Evaluation

Both models were evaluated on the held-out test split (314 samples)[cite: 64]:

### 1. Gaussian Naïve Bayes (GNB)
- Assumes normal distribution across independent features[cite: 64].
- Performance on test split[cite: 64]:
  - **Accuracy:** 45%[cite: 64]
  - **Minority Recall (`Rejected: 1`):** **67%** (catches 45 out of 67 potential loan rejections)[cite: 64]
  - **Minority Precision (`Rejected: 1`):** 23%[cite: 64]

### 2. K-Nearest Neighbors (KNN, $k=10$)
- Classifies instances based on Euclidean distance to the 10 nearest neighbors in standardized space[cite: 64].
- Performance on test split[cite: 64]:
  - **Accuracy:** **68%**[cite: 64]
  - **Majority Precision (`Approved: 0`):** 78%[cite: 64]
  - **Majority Recall (`Approved: 0`):** 82%[cite: 64]
  - **Minority Recall (`Rejected: 1`):** 16%[cite: 64]

### Confusion Matrix Breakdown (KNN)[cite: 64]
- **True Approved (TN):** 202[cite: 64]
- **False Rejected (FP):** 45[cite: 64]
- **False Approved (FN):** 56[cite: 64]
- **True Rejected (TP):** 11[cite: 64]

> **Insight:** GNB paired with SMOTE prioritizes risk mitigation (higher recall on rejections), while KNN provides higher overall prediction stability and accuracy across approved applicants[cite: 64].

---

## 🔮 Predicting for New Applicants

An inference template to format new data, align feature dummies, scale values, and generate predictions[cite: 64]:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Example single applicant record
new_applicant_data = {
    'age': 35,
    'annual_income': 600000,
    'loan_amount': 250000,
    'monthly_emi': 20000,
    'credit_score': 720,
    'existing_loans': 1,
    'has_property': 1,
    'city': 'Mumbai',
    'education': 'Graduate',
    'employment_type': 'Salaried'
}

# 1. Convert to DataFrame and apply one-hot encoding
new_df = pd.DataFrame([new_applicant_data])
new_processed = pd.get_dummies(new_df, columns=['city', 'education', 'employment_type'], dtype=int)

# 2. Align columns with training feature set (X)
missing_cols = set(X.columns) - set(new_processed.columns)
for c in missing_cols:
    new_processed[c] = 0
new_processed = new_processed[X.columns]

# 3. Scale and Predict
new_scaled = scaler.transform(new_processed)
prediction = neigh.predict(new_scaled)

status = "Rejected" if prediction[0] == 1 else "Approved"
print(f"Predicted Loan Status: {status}")

```

---

## 🚀 How to Run

### 1. Requirements

Install necessary dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

```

### 2. Execution

Run the analysis notebook via Jupyter:

```bash
jupyter notebook KNN_1.ipynb


