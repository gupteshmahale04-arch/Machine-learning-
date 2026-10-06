

# 🎓 Student Placement Prediction: Decision Tree Classifier

An end-to-end Machine Learning project demonstrating data cleaning, exploratory data analysis (EDA), categorical encoding, feature engineering, and classification modeling using a **Decision Tree Classifier** optimized with **RandomizedSearchCV**.

---

## 📌 Project Overview

College placement cells require early identification of students who may need additional academic mentoring, practice assessments, or career counseling before campus recruitment begins.

This project models placement readiness as a binary classification problem (`Placed` vs. `Not Placed`) by examining students' academic metrics, preparation habits, and demographic profiles. It details a complete machine learning workflow from raw, imperfect data through cleaning, baseline model benchmarking, hyperparameter tuning, and decision tree interpretability.

---

## 📊 Dataset Description

The dataset contains **10,000 student records** across **12 attributes**:

| Feature | Type | Description |
| --- | --- | --- |
| `student_id` | Identifier | Unique student identification code (non-predictive; dropped)|
| `gender` | Categorical | Student gender (`Male`, `Female`)|
| `branch` | Categorical | Academic department (`CSE`, `IT`, `ECE`, `Mechanical`, `Civil`, `Commerce`)|
| `study_hours` | Numerical | Daily self-study duration (hours/day)|
| `attendance` | Numerical | College lecture attendance percentage (0% – 100%)|
| `sleep_hours` | Numerical | Average sleep duration (hours/day)|
| `internet_usage` | Numerical | Daily non-academic internet usage (hours/day)|
| `assignments_completed` | Numerical | Total number of course assignments submitted (0 – 20)|
| `previous_score` | Numerical | Percentage scored in preceding academic terms|
| `extracurricular` | Categorical | Participation in extracurricular activities (`Yes` / `No`)|
| `exam_score` | Numerical | Score achieved on the placement assessment test (0 – 100)|
| `placement_status` | Categorical | **Target variable:** `Placed` or `Not Placed`<br> |

---

## 🛠️ Data Preprocessing & Feature Engineering Pipeline

1. **Deduplication:**
* Identified and removed **50 duplicate rows** from the original dataset.




2. **Missing Value Treatment:**
* Handled missing entries across academic and engagement fields (`attendance`, `study_hours`, `previous_score`, `assignments_completed`, etc.).


* Dropped incomplete rows, yielding **8,139 clean records**.




3. **Identifier Removal:**
* Dropped `student_id` to eliminate high-cardinality label noise without predictive power.




4. **Category Text Normalization:**
* Standardized casing and whitespace in `gender` (`M`, `male`, `F`, `female`) into uniform `male` and `female` categories.


* Standardized `branch` strings across inconsistent casing and whitespace variations.




5. **Outlier Mitigation:**
* Capped unrealistic recorded habit values using `.clip()` boundaries:


* `study_hours`: Capped within $[0, 12]$ hours


* `sleep_hours`: Capped within $[3, 12]$ hours


* `internet_usage`: Capped within $[0, 12]$ hours






6. **Feature Encoding:**
* Applied **One-Hot Encoding** (`pd.get_dummies`) to the multi-class `branch` feature.


* Applied `LabelEncoder` to binary variables: `gender`, `extracurricular`, and the target `placement_status`.





---

## ⚙️ Train-Test Split

The preprocessed dataset of **8,139 records** was partitioned using an 80/20 train-test ratio:

* **Training Set:** 6,511 samples


* **Testing Set:** 1,628 samples


* **Random Seed:** `random_state=42`


---

## 📈 Model Performance & Evaluation

The decision tree was benchmarked in its default state and subsequently optimized through 5-fold cross-validated hyperparameter search:

| Model Configuration | Criterion | Max Depth | Min Samples Leaf | Train Accuracy | Test Accuracy | Observations |
| --- | --- | --- | --- | --- | --- | --- |
| **Default Decision Tree** | `gini` | None (Unconstrained) | 1 | **100.0%** | **74.75%** | Overfitted training data by memorizing training paths.

 |
| **Tuned Decision Tree** | `entropy` | 7 | 2 | ~84.2% | **81.76%** | Generalizes significantly better on unseen test data.

 |

### Hyperparameter Search Space (`RandomizedSearchCV`)



```python
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

```

* **Best Parameters Found:** `{'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 2}`


> **Key Takeaway:** Unconstrained decision trees grow fully to achieve 100% training accuracy but suffer from high variance on held-out test data. Constraining maximum depth (`max_depth=7`) and minimum leaf size (`min_samples_leaf=2`) prunes overly specific splits and lifts holdout accuracy to **~81.8%**.
> 
> 

---

## 🚀 How to Run the Notebook

### 1. Prerequisites

Ensure you have Python 3.9+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib

```

### 2. Execution

Open and execute the notebook in JupyterLab or Google Colab:

```bash
jupyter notebook Decision_Tree.ipynb

```

---

## 📦 Key Dependencies

* **Pandas & NumPy:** Tabular data processing, deduplication, clipping, and numerical array handling.


* **Matplotlib & Seaborn:** Distribution boxplots, target count charts, and correlation heatmaps.


* **Scikit-Learn:** Preprocessing (`LabelEncoder`), data splitting (`train_test_split`), model fitting (`DecisionTreeClassifier`), hyperparameter optimization (`RandomizedSearchCV`), and evaluation metrics (`accuracy_score`).
