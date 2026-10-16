
# 🏦 Loan Default & Approval Prediction using Naïve Bayes

An end-to-end Machine Learning classification project designed to assess credit risk and predict loan default / approval status using demographic, financial, and credit history indicators[cite: 63].

---

## 📌 Project Overview

Financial institutions face significant risk when underwriting personal and consumer loans[cite: 63]. Evaluating creditworthiness accurately reduces default rates while maximizing viable loan originations[cite: 63]. 

This project implements a complete probabilistic classification pipeline using **Gaussian Naïve Bayes** to classify loan applications as either **Approved** or **Rejected** based on key applicant financial profiles[cite: 63].


```

Data Ingestion ➔ Data Cleaning ➔ EDA ➔ Feature Engineering ➔ Train/Test Split ➔ Model Training ➔ Evaluation ➔ Inference

```

---

## 📊 Dataset Overview

The dataset contains records of **1,000 loan applicants** across **12 attributes**[cite: 63]:

| Feature | Data Type | Description |
| :--- | :--- | :--- |
| `applicant_id` | Object | Unique applicant tracking code (e.g., `IND1000`)[cite: 63] |
| `age` | Integer | Applicant age in years[cite: 63] |
| `annual_income` | Integer | Total annual earnings in INR[cite: 63] |
| `loan_amount` | Integer | Requested loan principal amount[cite: 63] |
| `monthly_emi` | Float | Calculated monthly equated installment[cite: 63] |
| `credit_score` | Float | Credit bureau score (CIBIL/FICO range)[cite: 63] |
| `existing_loans` | Integer | Count of currently active credit lines[cite: 63] |
| `city` | Categorical | Metro/urban location (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune)[cite: 63] |
| `education` | Categorical | Highest educational qualification (`Graduate`, `Not Graduate`)[cite: 63] |
| `employment_type` | Categorical | Employment status (`Salaried`, `Self-Employed`, `Business Owner`, `Freelancer`)[cite: 63] |
| `has_property` | Binary | Asset ownership indicator (`1` = Yes, `0` = No)[cite: 63] |
| **`loan_status`** | **Categorical** | **Target variable (`Approved` / `Rejected`)**[cite: 63] |

---

## 🛠️ Data Preprocessing & Feature Engineering

1. **Handling Missing Values:**
   - Missing entries were identified in `monthly_emi` (20 rows) and `credit_score` (30 rows)[cite: 63].
   - Listwise deletion (`df.dropna(inplace=True)`) was executed, yielding a clean dataset of **950 observations**[cite: 63].

2. **Identifier Removal:**
   - `applicant_id` is an arbitrary non-predictive primary key[cite: 63]. Dropping this column avoids high-cardinality dimensionality explosion during one-hot encoding[cite: 63].

3. **Categorical Variable Encoding:**
   - Nominal predictors (`city`, `education`, `employment_type`) are transformed into dummy numerical features using `pd.get_dummies(..., drop_first=True)` to satisfy Gaussian Naïve Bayes requirements while avoiding the dummy variable trap[cite: 63].
   - Target variable `loan_status` is encoded numerically via `LabelEncoder` (`Approved = 0`, `Rejected = 1`)[cite: 63].

4. **Class Imbalance Consideration:**
   - The dataset exhibits skew toward approved applicants (~81% Approved vs. ~19% Rejected)[cite: 63].
   - Standard accuracy can be misleading in imbalanced contexts; performance is assessed through **Precision**, **Recall**, and **F1-Score**[cite: 63].

---

## ⚙️ Model Architecture & Methodology

- **Algorithm:** `GaussianNB` (Gaussian Naïve Bayes)[cite: 63].
- **Core Principle:** Applies Bayes' theorem under the conditional independence assumption between pairs of features given the class label:
  
  $$P(y \mid X) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(X)}$$

- **Distribution Assumption:** Features are assumed to follow a normal continuous bell curve $P(x_i \mid y) \sim \mathcal{N}(\mu_{y}, \sigma_{y}^2)$[cite: 63].
- **Validation Split:** 80% training set (760 samples) and 20% holdout test set (190 samples) with `random_state=42`[cite: 63].

---

## 📈 Evaluation & Results

Model performance evaluated on the 20% test partition:

| Metric | Target Class: Approved (0) | Target Class: Rejected (1) | Weighted Average |
| :--- | :---: | :---: | :---: |
| **Precision** | ~0.79 | ~0.25 | ~0.68 |
| **Recall** | ~0.95 | ~0.15 | ~0.79 |
| **F1-Score** | ~0.86 | ~0.19 | ~0.73 |
| **Overall Accuracy** | \multicolumn{3}{c|}{**~79.5%**} |

> 💡 **Key Finding:** In financial risk assessment, false negatives (approving a risky applicant who ends up defaulting) carry high cost. Setting balanced class priors (`GaussianNB(priors=[0.5, 0.5])`) or re-sampling (SMOTE) allows adjusting threshold boundaries to achieve higher rejection recall when risk aversion is prioritized.

---

## 🔮 Inference on New Applicant Data

The model supports live probability scoring for incoming loan applications:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# 1. Load and clean
df = pd.read_csv("loan_data (1).csv").dropna()

# 2. Separate target and features
y = LabelEncoder().fit_transform(df["loan_status"])  # 0: Approved, 1: Rejected
X = df.drop(columns=["applicant_id", "loan_status"])
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 3. Train-test split & model fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict on a new applicant profile
sample_applicant = pd.DataFrame([X_test.iloc[0]])
pred = model.predict(sample_applicant)[0]
prob = model.predict_proba(sample_applicant)[0]

print(f"Prediction: {'Rejected' if pred == 1 else 'Approved'}")
print(f"Approval Probability: {prob[0]*100:.2f}% | Default Risk: {prob[1]*100:.2f}%")

```

---

## 🚀 How to Run the Notebook

### 1. Requirements

Ensure Python 3.9+ is installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

```

### 2. Execution

Launch Jupyter Lab or Google Colab and run all cells sequentially:

```bash
jupyter notebook "loan_data (1).csv"

```

---

## 📦 Project Structure

```plaintext
├── loan_data (1).csv      # Loan dataset containing demographic and credit features
├── Loan_Prediction.ipynb  # Interactive development & model training notebook
└── README.md              # Project documentation and performance overview

