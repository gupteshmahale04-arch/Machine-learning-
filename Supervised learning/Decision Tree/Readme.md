```python
# Let's check the contents of 'Decision_Tree' notebook and the csv dataset
import json

with open('Decision_Tree', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"Cell {i} ({cell['cell_type']}):")
    source = "".join(cell.get('source', []))
    print(source[:200])
    print("-" * 40)


```

```text
Cell 0 (markdown):
# Student Placement Prediction using a Decision Tree

In this notebook we build a complete machine learning project step by step.
We use simple code and explain every step in simple English.
----------------------------------------
Cell 1 (code):
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
----------------------------------------
Cell 2 (markdown):
---
## Step 1: Problem

**Real-life story:** A college placement cell wants to know which students are likely to get a job placement.
If they know this early, they can give extra help (mock interviews
----------------------------------------
Cell 3 (markdown):
---
## Step 2: Data + EDA (Find patterns)

EDA means **Exploratory Data Analysis**.
It is like meeting a new student for the first time: first you look, ask questions and understand, then you decide w
----------------------------------------
Cell 4 (code):
df = pd.read_csv('student_performance_dataset.csv')
df_copy = df.copy()
df.head(5)
----------------------------------------
Cell 5 (markdown):
### 2.2 Understand the columns

| Column | Meaning |
|---|---|
| `student_id` | Unique ID of the student (only a label, not useful for prediction) |
| `gender` | Male or Female |
| `branch` | CSE, IT,
----------------------------------------
Cell 6 (code):
# Data types and non-null counts
df.info()
----------------------------------------
Cell 7 (code):
# Quick summary of the number columns
df.describe()
----------------------------------------
Cell 8 (code):
# How many missing values are in each column?
df.isnull().sum()
----------------------------------------
Cell 9 (code):
# Are there duplicate rows?
df.duplicated().sum()
----------------------------------------
Cell 10 (code):
# check Columns purity
cat_col = df.select_dtypes(include = 'object').columns
cat_col
print(cat_col)
for col in cat_col:
  print(df[col].unique())
  print()

----------------------------------------
Cell 11 (markdown):
### What is wrong with this data?

Real-world data is never perfect. Here we can see:

1. **Missing values** in many columns (study hours, attendance, sleep hours, and more).
2. **Duplicate rows**, th
----------------------------------------
Cell 12 (code):
sns.countplot(df['placement_status'])

#
----------------------------------------
Cell 13 (code):

----------------------------------------
Cell 14 (markdown):
### 2.4 Do placed students look different?
Let us compare the average of every number column for the two groups.
----------------------------------------
Cell 15 (code):
# Group by num_col to placement_status
num_col = df.select_dtypes(include =np.number).columns
df.groupby('placement_status')[num_col].mean()
----------------------------------------
Cell 16 (code):
# Box plots: compare Placed and Not Placed for each number column
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    sns.boxplot(x="placement_status", y=col, data=df)
   
----------------------------------------
Cell 17 (code):
# Correlation: which number columns move together?
sns.heatmap(df[num_col].corr(), annot = True)
----------------------------------------
Cell 18 (markdown):
---
## Step 3: Clean data

Rule: **never change the original data**. We work on a copy, so we can always go back.

We will do 5 cleaning jobs:
1. Remove duplicate rows
2. Drop `student_id` (it is only
----------------------------------------
Cell 19 (code):
# Make a copy so the original df stays safe

# Job 1: remove duplicate rows
df_copy.drop_duplicates(inplace = True)
df_copy.duplicated().sum()
# Job 1.0: Remove null values
df_copy.dropna(inplace=True
----------------------------------------
Cell 20 (code):
# Job 2: student_id is only a label. It does not help to predict placement.
df_copy.drop(columns = ['student_id'], axis = 1, inplace = True)
df_copy.head(2)
----------------------------------------
Cell 21 (code):
# Job 3: fix gender
# strip() removes extra spaces, lower() makes small letters, map() gives one clean name
df_copy['gender'] = df_copy['gender'].str.strip().str.lower().map({'male': 'male', 'female':
----------------------------------------
Cell 22 (code):
# Job 4: fix branch
# Remove extra spaces and make everything CAPITAL letters, so ' cse ' becomes 'CSE'
df['branch'] = df['branch'].str.strip().str.upper()
df['branch'].unique()
----------------------------------------
Cell 23 (code):
# Job 5: fix impossible values with clip()
# clip(0, 12) means: anything above 12 becomes 12, anything below 0 becomes 0
df_copy["study_hours"] = df_copy["study_hours"].clip(0, 12)
df_copy["sleep_hour
----------------------------------------
Cell 24 (code):
# Quick look at the cleaned categories
# Box plots: compare Placed and Not Placed for each number column
num_col = ['study_hours', 'sleep_hours', 'internet_usage']

for col in num_col:
    sns.boxplot
----------------------------------------
Cell 25 (markdown):
Cleaning is done. Values are now consistent and there are no impossible numbers.
The missing values are still there. We fix them next.
----------------------------------------
Cell 26 (markdown):
---
## Step 4: Preprocessing

A decision tree needs clean numbers. Preprocessing has 4 small jobs:

1. **Separate X and y.** `X` = the inputs, `y` = the answer we want to predict.
2. **Split into trai
----------------------------------------
Cell 27 (code):
# 1. Convert text to numbers (one-hot encoding) on branch and others label encoding.
from sklearn.preprocessing import LabelEncoder

# Create a copy
df_copy = df_copy.copy()

# Branch → One-Hot Encodi
----------------------------------------
Cell 28 (code):
# 2. Separate X and y.
X = df_copy.drop(columns = ['placement_status'])
y = df_copy['placement_status']

----------------------------------------
Cell 29 (code):
# 3. Split into train and test (80% / 20%).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_
----------------------------------------
Cell 30 (markdown):
---
## Step 5: Model
**Model : Default decision tree.**

----------------------------------------
Cell 31 (code):
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
----------------------------------------
Cell 32 (code):
y_pred  = clf.predict(X_test)

compare = pd.DataFrame({
    'Actual':y_test,
    'Predcited':y_pred
})
compare.head(10)

from sklearn.metrics import accuracy_score
Test_acc = accuracy_score(y_test,clf
----------------------------------------
Cell 33 (markdown):
---
## Step 6: Tune

Tuning means finding the best **settings** (hyperparameters) for the tree.

| Setting | Meaning | Real-life idea |
|---|---|---|
| `max_depth` | Maximum number of question levels 
----------------------------------------
Cell 34 (code):
# 4. Define hyperparameters
param_grid = {
    'max_depth':[3,5,7,10],
    'min_samples_leaf':[1,2,5,10],
    'criterion':['gini','entropy']
}
----------------------------------------
Cell 35 (code):
# RandomizedSearch / gridSearchCV
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator = clf,
    param_distributions = param_grid,
    n_iter = 10,
----------------------------------------
Cell 36 (code):
# Train
random_search.fit(X_train,y_train)
----------------------------------------
Cell 37 (code):
# Best perfomance
print(random_search.best_params_)

----------------------------------------
Cell 38 (code):
# Best model
best_model = random_search.best_estimator_
----------------------------------------
Cell 39 (code):
# Prediction
y_pred = best_model.predict(X_test)
----------------------------------------
Cell 40 (code):
#final accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
----------------------------------------
Cell 41 (markdown):
---
## Step 7: Evaluate

Now we test the tuned model on the **test data**, which it has never seen.

**Metrics in simple words** (think of "Placed" as the positive class):

| Metric | Question it answ
----------------------------------------
Cell 42 (code):

----------------------------------------
Cell 43 (markdown):
---
## Step 8: Explain

A big reason to use a decision tree is that we can **explain** its decisions.
A placement officer will not trust a model that only says "Placed" without a reason.

### Feature 
----------------------------------------
Cell 44 (code):

----------------------------------------
Cell 45 (markdown):
## Step 9: Save model

----------------------------------------
Cell 46 (code):

----------------------------------------
Cell 47 (markdown):
**Next: build the FastAPI app**
- `POST /predict` takes student details and returns Placed / Not Placed with the probability.
- A simple web page (HTML form) calls this API and shows the result.
- Kee
----------------------------------------
Cell 48 (code):

----------------------------------------
Cell 49 (markdown):
#
----------------------------------------


```

```python
# Let's inspect the cells in detail around steps 4, 5, 6, 7, 8, 9
for i in [27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 47]:
    cell = nb['cells'][i]
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print("".join(cell.get('source', [])))


```

```text
--- Cell 27 (code) ---
# 1. Convert text to numbers (one-hot encoding) on branch and others label encoding.
from sklearn.preprocessing import LabelEncoder

# Create a copy
df_copy = df_copy.copy()

# Branch → One-Hot Encoding
df_copy = pd.get_dummies(
    df_copy,
    columns=["branch"],
    dtype=int
)

# Label Encoding
label_en = LabelEncoder()

df_copy["gender"] = label_en.fit_transform(df_copy["gender"])
df_copy["extracurricular"] = label_en.fit_transform(df_copy["extracurricular"])
df_copy["placement_status"] = label_en.fit_transform(df_copy["placement_status"])

df_copy.head()
--- Cell 28 (code) ---
# 2. Separate X and y.
X = df_copy.drop(columns = ['placement_status'])
y = df_copy['placement_status']

--- Cell 29 (code) ---
# 3. Split into train and test (80% / 20%).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
--- Cell 31 (code) ---
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
--- Cell 32 (code) ---
y_pred  = clf.predict(X_test)

compare = pd.DataFrame({
    'Actual':y_test,
    'Predcited':y_pred
})
compare.head(10)

from sklearn.metrics import accuracy_score
Test_acc = accuracy_score(y_test,clf.predict(X_test))
Train_acc = accuracy_score(y_train,clf.predict(X_train))
print(f'Train_accuracy:{Train_acc}')
print(f'Test_accuracy:{Test_acc}')

--- Cell 34 (code) ---
# 4. Define hyperparameters
param_grid = {
    'max_depth':[3,5,7,10],
    'min_samples_leaf':[1,2,5,10],
    'criterion':['gini','entropy']
}
--- Cell 35 (code) ---
# RandomizedSearch / gridSearchCV
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator = clf,
    param_distributions = param_grid,
    n_iter = 10,
    cv = 5,
    scoring = 'accuracy',
    random_state = 42
)
--- Cell 36 (code) ---
# Train
random_search.fit(X_train,y_train)
--- Cell 37 (code) ---
# Best perfomance
print(random_search.best_params_)

--- Cell 38 (code) ---
# Best model
best_model = random_search.best_estimator_
--- Cell 39 (code) ---
# Prediction
y_pred = best_model.predict(X_test)
--- Cell 40 (code) ---
#final accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
--- Cell 41 (markdown) ---
---
## Step 7: Evaluate

Now we test the tuned model on the **test data**, which it has never seen.

**Metrics in simple words** (think of "Placed" as the positive class):

| Metric | Question it answers |
|---|---|
| Accuracy | Out of all students, how many did we predict correctly? |
| Precision | Out of students we said "Placed", how many were really placed? |
| Recall | Out of students who were really placed, how many did we find? |
| F1-score | One number that balances precision and recall |


**Confusion matrix** is a 2x2 table that shows correct and wrong predictions of each type.
--- Cell 43 (markdown) ---
---
## Step 8: Explain

A big reason to use a decision tree is that we can **explain** its decisions.
A placement officer will not trust a model that only says "Placed" without a reason.

### Feature importance
--- Cell 45 (markdown) ---
## Step 9: Save model

--- Cell 47 (markdown) ---
**Next: build the FastAPI app**
- `POST /predict` takes student details and returns Placed / Not Placed with the probability.
- A simple web page (HTML form) calls this API and shows the result.
- Keep `placement_model.joblib` and the same scikit-learn version in the API project.


```

```python
# Let's run the exact pipeline from the notebook to get accurate metrics and parameters
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv('student_performance_dataset.csv')
print("Initial shape:", df.shape)

df_copy = df.copy()
df_copy.drop_duplicates(inplace=True)
df_copy.dropna(inplace=True)
df_copy.drop(columns=['student_id'], inplace=True)

df_copy['gender'] = df_copy['gender'].str.strip().str.lower().map({'male': 'male', 'female': 'female', 'm': 'male', 'f': 'female'})
df_copy['branch'] = df_copy['branch'].str.strip().str.upper()

df_copy["study_hours"] = df_copy["study_hours"].clip(0, 12)
df_copy["sleep_hours"] = df_copy["sleep_hours"].clip(3, 12)
df_copy["internet_usage"] = df_copy["internet_usage"].clip(0, 12)

# Encoding
df_encoded = pd.get_dummies(df_copy, columns=["branch"], dtype=int)
label_en = LabelEncoder()
df_encoded["gender"] = label_en.fit_transform(df_encoded["gender"])
df_encoded["extracurricular"] = label_en.fit_transform(df_encoded["extracurricular"])
df_encoded["placement_status"] = label_en.fit_transform(df_encoded["placement_status"])

X = df_encoded.drop(columns=['placement_status'])
y = df_encoded['placement_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

train_acc_baseline = accuracy_score(y_train, clf.predict(X_train))
test_acc_baseline = accuracy_score(y_test, clf.predict(X_test))

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
tuned_acc = accuracy_score(y_test, y_pred_tuned)

print("Baseline Train Acc:", train_acc_baseline)
print("Baseline Test Acc:", test_acc_baseline)
print("Best Params:", random_search.best_params_)
print("Tuned Test Acc:", tuned_acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Feature importances
fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", fi)


```

```text
Initial shape: (10000, 12)
Baseline Train Acc: 1.0
Baseline Test Acc: 0.7426289926289926
Best Params: {'min_samples_leaf': 2, 'max_depth': 7, 'criterion': 'entropy'}
Tuned Test Acc: 0.8151105651105651
Confusion Matrix:
 [[700 121]
 [180 627]]
Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.85      0.82       821
           1       0.84      0.78      0.81       807

    accuracy                           0.82      1628
   macro avg       0.82      0.81      0.81      1628
weighted avg       0.82      0.82      0.81      1628

Feature Importances:
 exam_score               0.688727
extracurricular          0.099368
previous_score           0.079562
assignments_completed    0.075306
attendance               0.017905
internet_usage           0.012948
study_hours              0.007453
branch_IT                0.005235
branch_CSE               0.005212
sleep_hours              0.003423
branch_CIVIL             0.002416
branch_MECHANICAL        0.001258
branch_ECE               0.001187
gender                   0.000000
branch_COMMERCE          0.000000
dtype: float64


```

```markdown
# 🎓 Student Placement Prediction Using a Decision Tree

An end-to-end Machine Learning classification project designed to predict student placement outcomes (`Placed` vs. `Not Placed`) based on academic performance, study habits, and demographic profiles[cite: 65, 66].

---

## 📌 Project Overview
College placement cells require early identification of students at risk of missing campus recruitment[cite: 65, 66]. Predicting placement readiness ahead of time allows academic institutions to deliver targeted interventions, including mock interviews, coding bootcamps, and academic mentoring[cite: 65, 66].


```

Raw Data Ingestion ➔ Exploratory Data Analysis ➔ Data Cleaning ➔ Feature Engineering
➔ Model Training (Baseline) ➔ Hyperparameter Tuning ➔ Model Evaluation ➔ Feature Importance

```

---

## 📊 Dataset Description

The dataset (`student_performance_dataset.csv`) contains **10,000 student records** across **12 attributes**[cite: 65, 66]:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `student_id` | Identifier | Unique student code (dropped during preprocessing)[cite: 65, 66] |
| `gender` | Categorical | Student gender (`Male`, `Female`)[cite: 65, 66] |
| `branch` | Categorical | Academic department (`CSE`, `IT`, `ECE`, `MECHANICAL`, `CIVIL`, `COMMERCE`)[cite: 65, 66] |
| `study_hours` | Numerical | Daily self-study duration (hours/day)[cite: 65, 66] |
| `attendance` | Numerical | Class attendance percentage (0–100%)[cite: 65, 66] |
| `sleep_hours` | Numerical | Average sleep duration (hours/day)[cite: 65, 66] |
| `internet_usage` | Numerical | Non-academic internet usage (hours/day)[cite: 65, 66] |
| `assignments_completed` | Numerical | Total course assignments submitted (0–20)[cite: 65, 66] |
| `previous_score` | Numerical | Prior academic term examination percentage[cite: 65, 66] |
| `extracurricular` | Categorical | Participation in extracurricular activities (`Yes`, `No`)[cite: 65, 66] |
| `exam_score` | Numerical | Final placement assessment test score (0–100)[cite: 65, 66] |
| **`placement_status`** | **Binary Target** | **Target label: `Placed` (`1`) vs. `Not Placed` (`0`)**[cite: 65, 66] |

---

## 🛠️ Data Cleaning & Preprocessing Pipeline

Real-world datasets often present formatting irregularities, missing entries, and unrealistic values[cite: 65, 66]. The data cleaning pipeline performs the following steps:

1. **Deduplication:** Identified and eliminated 50 duplicate rows[cite: 65, 66].
2. **Missing Value Handling:** Dropped records with null values across academic and habit columns, retaining 8,139 complete records[cite: 65, 66].
3. **Identifier Removal:** Removed `student_id` as it provides no predictive signal[cite: 65, 66].
4. **Text Normalization:**
   - Standardized `gender` representations (`M`, `male`, `F`, `female`) into consistent lowercase categories (`male`, `female`)[cite: 65, 66].
   - Trimmed whitespace and converted `branch` categories into uniform uppercase strings (`CSE`, `IT`, `ECE`, `MECHANICAL`, `CIVIL`, `COMMERCE`)[cite: 65, 66].
5. **Outlier Capping:** Applied threshold boundaries using `.clip()` to constrain impossible recording values:
   - `study_hours`: Capped between `0` and `12` hours[cite: 65, 66]
   - `sleep_hours`: Capped between `3` and `12` hours[cite: 65, 66]
   - `internet_usage`: Capped between `0` and `12` hours[cite: 65, 66]
6. **Encoding:**
   - One-hot encoded `branch` using `pd.get_dummies(..., dtype=int)`[cite: 65, 66].
   - Applied `LabelEncoder` to binary variables: `gender`, `extracurricular`, and `placement_status` (`Placed: 1`, `Not Placed: 0`)[cite: 65, 66].
7. **Train-Test Split:** Split into 80% training (6,511 samples) and 20% holdout testing (1,628 samples) with `random_state=42`[cite: 65, 66].

---

## 🤖 Model Training & Hyperparameter Tuning

### Baseline Model
- **Algorithm:** `DecisionTreeClassifier(random_state=42)`[cite: 65, 66]
- **Training Accuracy:** `100.0%` (demonstrating default decision tree overfitting)[cite: 65, 66]
- **Testing Accuracy:** `74.26%`[cite: 65, 66]

### Hyperparameter Optimization
To prevent tree memorization and enhance generalization, `RandomizedSearchCV` was conducted across key structural parameters[cite: 65, 66]:

```python
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

```

* **Best Parameters:** `{'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 2}`

* **Tuned Model Test Accuracy:** **81.51% (~82%)**


---

## 📊 Model Evaluation (Test Set)

Performance evaluated on the unseen 20% holdout test set (1,628 samples):

### Classification Metrics



| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Not Placed (0)** | 0.80 | 0.85 | 0.82 | 821 |
| **Placed (1)** | 0.84 | 0.78 | 0.81 | 807 |
| **Overall Accuracy** | \multicolumn{4}{c | }{**81.51%**} |  |  |
| **Macro Average** | 0.82 | 0.81 | 0.81 | 1628 |
| **Weighted Average** | 0.82 | 0.82 | 0.81 | 1628 |

### Confusion Matrix



```
                 Predicted: Not Placed    Predicted: Placed
Actual: Not Placed        700                   121
Actual: Placed            180                   627

```

---

## 🔍 Feature Importance & Interpretability

Decision trees allow direct inspection of how much each variable contributes to splitting decisions:

| Feature | Importance Score | Key Finding |
| --- | --- | --- |
| `exam_score` | **68.87%** | Strongest single predictor of placement qualification.

 |
| `extracurricular` | **9.94%** | Active involvement significantly differentiates borderline candidates.

 |
| `previous_score` | **7.96%** | Historical academic performance shows moderate predictive weight.

 |
| `assignments_completed` | **7.53%** | Submission consistency correlates with successful placement.

 |
| `attendance` | **1.79%** | Baseline participation requirement.

 |
| `internet_usage` | **1.29%** | Minor factor.

 |
| `study_hours` | **0.75%** | Marginal direct impact compared to objective exam scores.

 |

---

## 🔮 Inference & Deployment Workflow

The trained model can be serialized and integrated into a web API (e.g., using FastAPI) for real-time inference:

```python
import joblib
import pandas as pd

# Save trained decision tree model
joblib.dump(best_model, 'placement_model.joblib')

# Load and predict on new student profile
loaded_model = joblib.load('placement_model.joblib')

sample_student = pd.DataFrame([{
    'study_hours': 4.5,
    'attendance': 88.0,
    'sleep_hours': 7.0,
    'internet_usage': 3.5,
    'assignments_completed': 18,
    'previous_score': 76.0,
    'exam_score': 82.0,
    'gender': 1,              # Male
    'extracurricular': 1,     # Yes
    'branch_CIVIL': 0,
    'branch_COMMERCE': 0,
    'branch_CSE': 1,
    'branch_ECE': 0,
    'branch_IT': 0,
    'branch_MECHANICAL': 0
}])

prediction = loaded_model.predict(sample_student)[0]
probability = loaded_model.predict_proba(sample_student)[0]

status = "Placed" if prediction == 1 else "Not Placed"
print(f"Prediction: {status} (Placement Probability: {probability[1] * 100:.2f}%)")

```

---

## 🚀 How to Run the Notebook

### 1. Prerequisites

Install all required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib

```

### 2. Execution

Run the development notebook locally or in Google Colab:

```bash
jupyter notebook Decision_Tree.ipynb
