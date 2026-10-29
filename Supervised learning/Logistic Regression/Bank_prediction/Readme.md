

# 🏦 Bank Term Deposit Subscription Prediction: Logistic Regression Pipeline

An end-to-end Machine Learning classification project focused on predicting whether a banking client will subscribe to a term deposit (`yes` / `no`) based on demographic profiles, previous campaign interactions, call metrics, and macroeconomic indicators.

---

## 📌 Project Overview

Direct telemarketing campaigns require significant operational resources. Financial institutions face high contact volumes with low baseline conversion rates (~11.3% in this portfolio).

This project formulates deposit subscription as an imbalanced binary classification problem. By employing **cost-sensitive learning (`class_weight='balanced'`)** within a **Logistic Regression** framework, the model prioritizes **Recall (87%)** to minimize false negatives—ensuring the sales team rarely misses a high-propensity client.

---

## 📊 Dataset Description

The dataset (`bank_additional_full_structured.csv`) contains **41,188 customer records** across **21 features**:

| Feature Category | Features | Description |
| --- | --- | --- |
| **Client Demographics** | `age`, `job`, `marital`, `education` | Customer age, employment sector, marital status, and completed education level.

 |
| **Credit & Account History** | `default`, `housing`, `loan` | Indicators of credit default, existing housing loans, and personal loans.

 |
| **Direct Marketing Contact** | `contact`, `month`, `day_of_week`, `duration` | Communication type (cellular / telephone), last contact timing, and call duration in seconds.

 |
| **Campaign History** | `campaign`, `pdays`, `previous`, `poutcome` | Number of contacts during current campaign, days elapsed since last contact, previous contacts count, and outcome of past campaigns.

 |
| **Macroeconomic Indicators** | `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed` | Employment variation rate, consumer price index, consumer confidence index, 3-month Euribor rate, and number of employees.

 |
| **Target Variable** | **`y`** | Binary outcome: Has the client subscribed to a term deposit? (`yes`: 1, `no`: 0).

 |

---

## 🛠️ Data Preprocessing & Cleaning Pipeline

1. **Deduplication:**
* Identified and removed **12 duplicate records** from the initial 41,188 rows.




2. **Outlier Filtering (Domain Capping):**
* Conducted boxplot analysis across numerical distributions (`age`, `duration`, `campaign`, `previous`, `cons_conf_idx`).


* Filtered out extreme senior age outliers by enforcing an upper limit:

$$\text{age} \le 80$$


* Retained **41,057 high-quality records**.




3. **Encoding Categorical Features:**
* Evaluated categorical columns: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`, and target `y`.


* Mapped categorical levels into numerical representations using Scikit-Learn's `LabelEncoder`.




4. **Feature Standardization:**
* Because Logistic Regression computes linear decision boundaries sensitive to variable variance, all 20 predictive features were centered and scaled:

$$z = \frac{x - \mu}{\sigma}$$


* Fit `StandardScaler` on the training set and transformed the held-out test set to prevent data leakage.





---

## ⚙️ Train-Test Split & Validation Strategy

The cleaned dataset of **41,057 samples** was split into an 80/20 train-test partition:

* **Training Set:** 32,845 records


* **Testing Set:** 8,212 records


* **Cross-Validation:** 10-Fold Stratified Cross-Validation on the training subset


* **Random State:** `random_state=42`



---

## 📈 Model Performance & Evaluation

The model was trained with balanced class weights to penalize under-predicting the minority subscription class:

| Metric | Score | Key Takeaway |
| --- | --- | --- |
| **10-Fold CV Recall** | **87.20%** | Cross-validation demonstrates robust generalization on positive identifications.

 |
| **Test Accuracy** | **85.00%** | Strong overall correctness across all predictions.

 |
| **Test Recall** | **87.00%** | Successfully captures ~87 out of every 100 actual deposit subscribers.

 |
| **Test Precision** | **42.00%** | Expected trade-off: higher false positives accepted to prevent missed leads.

 |
| **Test F1-Score** | **0.56** | Harmonic balance between precision and prioritized recall.

 |

> **Business Takeaway:** In high-value telemarketing, the cost of reaching out to a false lead is far lower than the opportunity cost of losing an actual deposit customer. By tuning the classifier to achieve **87% Recall**, marketing teams capture the vast majority of conversions.
> 
> 

---

## 🔮 Real-Time Client Scoring & Inference

Incoming customer profiles can be standardized and evaluated for real-time subscription probability:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Example new customer payload
new_customer = pd.DataFrame([{
    "age": 50,
    "job": 0,           # admin.
    "marital": 1,       # married
    "education": 6,     # university.degree
    "default": 0,       # no
    "housing": 2,       # yes
    "loan": 0,          # no
    "contact": 0,       # cellular
    "month": 6,         # may
    "day_of_week": 1,   # mon
    "duration": 180,    # average call duration (seconds)
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": 1,      # nonexistent
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
}])

# Scale using trained scaler and predict
new_customer_scaled = scaler.transform(new_customer)
prediction = clf.predict(new_customer_scaled)
probability = clf.predict_proba(new_customer_scaled)[0]

print(f"Prediction: {'Subscribe (1)' if prediction[0] == 1 else 'No Subscription (0)'}")
print(f"Confidence: No Subscription = {probability[0]*100:.1f}% | Subscription = {probability[1]*100:.1f}%")

```

---

## 🚀 How to Run the Notebook

### 1. Prerequisites

Ensure Python 3.9+ is installed along with the required scientific packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

```

### 2. Execution

Run the analysis notebook in JupyterLab, VS Code, or Google Colab:

```bash
jupyter notebook Logistic_Regression_.ipynb

```

---

## 📦 Key Dependencies

* **Pandas & NumPy:** In-memory tabular operations, data filtering, slicing, and matrix transformations.


* **Matplotlib & Seaborn:** Outlier detection boxplots and exploratory distribution visuals.


* **Scikit-Learn:** Preprocessing (`LabelEncoder`, `StandardScaler`), model training (`LogisticRegression`), model selection (`train_test_split`, `cross_val_score`), and classification metrics (`recall_score`, `precision_score`, `accuracy_score`, `f1_score`).
