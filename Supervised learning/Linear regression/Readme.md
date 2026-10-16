
# 🚗 Used Car Price Prediction: Linear, Ridge & Lasso Regression

An end-to-end Machine Learning project demonstrating data cleaning, exploratory data analysis (EDA), outlier treatment, skewness transformation, and comparative evaluation using **Linear Regression**, **Ridge Regression (L2 Regularization)**, and **Lasso Regression (L1 Regularization)**.

---

## 📌 Project Overview

Predicting the market value of a used car involves handling non-linear interactions across vehicle specifications (brand, body type, mileage, engine volume, fuel type, registration status, and production year). 

This project explores the regression workflow from raw, messy real-world data to production evaluation metrics ($R^2$), illustrating why regularization and log-transforms are essential for predictive modeling.

---

## 📊 Dataset Description

The dataset consists of **4,345 records** across **9 features** sourced from used car sales data:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Brand` | Categorical | Manufacturer (BMW, Mercedes-Benz, Audi, Toyota, Volkswagen, etc.) |
| `Price` | Numerical | Target variable representing sales price (USD) |
| `Body` | Categorical | Vehicle body style (sedan, van, crossover, hatch, etc.) |
| `Mileage` | Numerical | Total distance traveled (in thousands of km) |
| `EngineV` | Numerical | Engine volume/displacement in liters |
| `Engine Type` | Categorical | Fuel type (Diesel, Petrol, Gas, Other) |
| `Registration`| Categorical | Registration status (`yes` / `no`) |
| `Year` | Numerical | Year of production (1969 – 2016) |
| `Model` | Categorical | Specific vehicle model name (dropped due to high cardinality) |

---

## 🛠️ Data Preprocessing & Feature Engineering Pipeline

1. **Handling Missing Values:**
   - Identified missing values in `Price` and `EngineV`.
   - Used `df.dropna(subset=['Price', 'EngineV'])` to drop rows with unrecoverable target and core technical values.

2. **Domain-Driven Outlier Detection:**
   - Detected unrealistic engine capacities (e.g., maximum recorded `EngineV = 99.99` L due to entry typos).
   - Filtered dataset to retain realistic consumer automotive engine displacements:
     $$\text{EngineV} \le 10.0\text{ Liters}$$

3. **Target Transformation (Log Transformation):**
   - The original `Price` variable exhibited strong positive skewness.
   - Applied natural logarithm transformation:
     `$$\text{Log\_price} = \ln(\text{Price})$$`
   - Reduced heteroscedasticity and aligned the target distribution closer to normality.

4. **Dimensionality & High-Cardinality Management:**
   - Dropped `Model` (300+ unique classes) to avoid sparse matrices and excessive dummy variables.
   - Dropped original `Price` after log-transformation.

5. **Encoding Categorical Features:**
   - Applied `LabelEncoder` across all remaining categorical columns (`Brand`, `Body`, `Engine Type`, `Registration`).

---

## ⚙️ Train-Test Split

The final dataset of **4,005 samples** and **7 features** was split into training and testing partitions using an 80/20 ratio:

- **Training Set:** 3,204 samples
- **Testing Set:** 801 samples
- **Random Seed:** `random_state=42`

---

## 📈 Model Performance & Evaluation

All three regression models were evaluated on the held-out test partition using the Coefficient of Determination ($R^2$ score):

| Model | Technique | Test $R^2$ Score | Observations |
| :--- | :--- | :---: | :--- |
| **Linear Regression** | Ordinary Least Squares (OLS) | **0.8296** | Baseline model fitting the linear relationship. |
| **Ridge Regression** | L2 Regularization ($\alpha=1.0$) | **0.8296** | Shrinks coefficients; prevents multicollinearity without zeroing coefficients. |
| **Lasso Regression** | L1 Regularization ($\alpha=1.0$) | **0.5139** | Underperformed with default parameters; aggressive sparsity penalty zeroed out key predictors. |

> **Key Takeaway:** Standard OLS and Ridge Regression both explain **~83% of the variance** in vehicle log prices. Default Lasso regression penalized the feature weights too heavily, indicating hyperparameter tuning ($\alpha$) or feature scaling (`StandardScaler`) is necessary for L1 regularization.

---

## 🚀 How to Run the Notebook

### 1. Prerequisites
Ensure you have Python 3.9+ installed along with the required scientific computing libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub

```

### 2. Execution

Open and run the notebook in JupyterLab or Google Colab:

```bash
jupyter notebook Car_Price_Prediction.ipynb

```

---

## 📦 Key Dependencies

* **Pandas & NumPy:** Data wrangling, linear indexing, and mathematical operations.
* **Matplotlib & Seaborn:** Distribution histograms, boxplots, correlation heatmaps, and regression residual scatterplots.
* **Scikit-Learn:** Data preprocessing (`LabelEncoder`, `train_test_split`), regression estimators (`LinearRegression`, `Ridge`, `Lasso`), and model evaluation (`r2_score`).
