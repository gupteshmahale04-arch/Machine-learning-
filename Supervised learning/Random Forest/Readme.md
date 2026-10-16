# 🛵 Food Delivery Time Prediction: Regression Pipeline with Decision Tree & Random Forest

An end-to-end Machine Learning regression project demonstrating exploratory data analysis (EDA), target transformation, pipeline-based preprocessing with scikit-learn (`Pipeline` & `ColumnTransformer`), and comparative modeling using **Decision Tree Regressor** and **Random Forest Regressor**.

---

## 📌 Project Overview

Accurately predicting food delivery times is essential for on-demand logistics, customer satisfaction, and courier dispatch optimization. Delivery durations depend on multiple non-linear interactions across spatial, environmental, vehicular, and operational parameters (such as transit distance, weather, traffic, vehicle type, and kitchen preparation time).

This project implements an automated, leakage-free preprocessing and modeling workflow using scikit-learn pipelines to predict food delivery durations (in minutes) and evaluates model performance across tree-based algorithms.

---

## 📊 Dataset Description

The dataset (`Food_Delivery_Times.csv`) consists of **1,000 delivery records** across **9 features**:

| Feature | Type | Description |
| --- | --- | --- |
| `Order_ID` | Identifier | Unique order ID (dropped before model training) |
| `Distance_km` | Numerical | Delivery distance between restaurant and customer (0.59 – 19.99 km) |
| `Weather` | Categorical | Weather conditions (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Windy`) |
| `Traffic_Level` | Categorical | Traffic congestion level (`Low`, `Medium`, `High`) |
| `Time_of_Day` | Categorical | Shift / order time (`Morning`, `Afternoon`, `Evening`, `Night`) |
| `Vehicle_Type` | Categorical | Delivery vehicle mode (`Bike`, `Scooter`, `Car`) |
| `Preparation_Time_min` | Numerical | Food preparation/kitchen packaging time in minutes (5 – 29 min) |
| `Courier_Experience_yrs` | Numerical | Experience level of the courier in years (0.0 – 9.0 yrs) |
| **`Delivery_Time_min`** | **Numerical (Target)** | Total duration required to complete the delivery (8 – 153 min) |

---

## 🛠️ Data Preprocessing & Pipeline Engineering

1. **Identifier Removal:**
* `Order_ID` was removed from the feature matrix $X$ as it provides no predictive value and risks model memorization.


2. **Target Skewness Correction (Log-Transformation):**
* The raw target `Delivery_Time_min` showed positive skewness ($\approx +0.51$) with a right-tailed distribution.
* Applied natural logarithm transformation to stabilize variance and normalize residuals:

$$\text{Target} = \ln(\text{Delivery\_Time\_min})$$


* Predictions are mapped back to interpretable minutes during evaluation via exponential transformation ($\exp(\hat{y})$).


3. **Leakage-Free Preprocessing Pipeline:**
* Preprocessing steps were organized using `ColumnTransformer` to ensure test and inference data do not leak into training statistics:
* **Categorical Pipeline (`cat_cols`):**
* Imputation via `SimpleImputer(strategy='most_frequent')` to fill missing entries in `Weather`, `Traffic_Level`, and `Time_of_Day` (30 nulls each).
* Encoding via `OneHotEncoder(handle_unknown='ignore')` to convert categories into binary flags without breaking on unseen levels.


* **Numerical Pipeline (`num_cols`):**
* Imputation via `SimpleImputer(strategy='median')` to handle missing values in `Courier_Experience_yrs` (30 nulls).







---

## ⚙️ Train-Test Split

The dataset was partitioned into training and evaluation sets using an 80/20 split:

* **Training Set:** 800 orders
* **Testing Set:** 200 orders
* **Random Seed:** `random_state=42`

---

## 📈 Model Performance & Evaluation

Both models were integrated directly into the scikit-learn pipeline (`prep` + `model`) and evaluated on the held-out test split using $R^2$ (Coefficient of Determination) score:

| Model | Hyperparameters | Train $R^2$ Score | Test $R^2$ Score | Observations |
| --- | --- | --- | --- | --- |
| **Decision Tree Regressor** | `max_depth=5`, `random_state=42` | **0.8252** | **0.7480** | Interpretable rule-based baseline; suffers from slight variance and step-wise predictions. |
| **Random Forest Regressor** | `n_estimators=100`, `random_state=42` | **0.9710** | **0.8481** | Ensemble of 100 de-correlated trees; significantly reduces prediction error and explains **~85% of test variance**. |

> **Key Takeaway:** Random Forest out-performs the single Decision Tree by **+10% on test $R^2$**, effectively averaging out variance across individual trees and modeling continuous delivery durations with higher fidelity.

---

## 🔮 Inference on Unseen Orders (Pipeline Advantage)

Because the model and preprocessing are unified inside a single `Pipeline`, incoming orders can be fed as raw DataFrames with missing attributes without manual encoding or manual imputation steps:

```python
import numpy as np
import pandas as pd

# Define a brand new incoming order with raw attributes and missing courier experience
new_order = pd.DataFrame([{
    'Distance_km': 8.5,
    'Weather': 'Rainy',
    'Traffic_Level': 'High',
    'Time_of_Day': 'Evening',
    'Vehicle_Type': 'Scooter',
    'Preparation_Time_min': 15,
    'Courier_Experience_yrs': None  # Handled automatically via median imputer
}])

# Predict log time and transform back to actual minutes
predicted_log_time = rf_model.predict(new_order)[0]
predicted_time_min = np.exp(predicted_log_time)

print(f"Predicted Delivery Time: {predicted_time_min:.1f} minutes")
# Output: Predicted delivery time for the new order: 51.9 minutes

```

---

## 🚀 How to Run the Notebook

### 1. Prerequisites

Ensure Python 3.9+ is installed along with the required scientific computing libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

```

### 2. Execution

Open and run the notebook in JupyterLab, VS Code, or Google Colab:

```bash
jupyter notebook Food_Delivery_Times.ipynb

```

---

## 📦 Key Dependencies

* **Pandas & NumPy:** Data wrangling, log/exponential transformations, and array operations.
* **Matplotlib & Seaborn:** Distribution histograms, KDE skewness plots, and residual visualizations.
* **Scikit-Learn:** Preprocessing (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`), regression modeling (`DecisionTreeRegressor`, `RandomForestRegressor`, `RandomizedSearchCV`), and metrics (`r2_score`, `mean_squared_error`, `mean_absolute_error`).
