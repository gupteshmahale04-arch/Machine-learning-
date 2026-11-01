# House Price Prediction using Machine Learning

A machine learning project that explores residential property data and predicts house prices using regression models. The analysis and model experiments are documented in the Jupyter Notebook.

## Project Overview

The notebook uses property characteristics such as location, property type, size, number of rooms, furnishing, amenities, and distance from key locations to estimate the property's price.

The target variable is `Price_INR_Lakhs`. In the notebook, the target is transformed using the natural logarithm before model training:

```python
df["log_Price_INR_Lakhs"] = np.log(df["Price_INR_Lakhs"])
```

The notebook experiments with:
- Linear Regression
- Ridge Regression
- Lasso Regression

Models are compared using the **R-squared (R²)** metric.

## Dataset

The supplied dataset contains **80,000 rows and 18 columns**.

| Column | Description |
|---|---|
| `Property_ID` | Unique property identifier (removed before modelling) |
| `City` | City in which the property is located |
| `Locality_Type` | Locality category |
| `Property_Type` | Type of property |
| `BHK` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Super_Area_SqFt` | Super built-up area in square feet |
| `Carpet_Area_SqFt` | Carpet area in square feet |
| `Floor_Number` | Floor on which the property is located |
| `Total_Floors` | Total floors in the building |
| `Age_of_Property` | Property age |
| `Furnishing_Status` | Furnishing category |
| `Parking` | Parking availability/value as represented in the dataset |
| `Lift_Available` | Lift availability indicator |
| `Gated_Community` | Gated community indicator |
| `Distance_to_Metro_km` | Distance from metro in kilometres |
| `Distance_to_City_Center_km` | Distance from city centre in kilometres |
| `Price_INR_Lakhs` | Property price in INR lakhs; prediction target |

## Project Workflow

1. **Load the data** with Pandas and create a working copy.
2. **Explore the dataset** using `head()`, `tail()`, `shape`, `info()`, and `describe()`.
3. **Check data quality** by checking missing values and duplicate rows.
4. **Prepare the target** by removing `Property_ID`, taking the natural log of the price, and using the transformed target for modelling.
5. **Perform exploratory data analysis (EDA)** with histograms, scatter plots, bar plots, and a correlation heatmap.
6. **Encode categorical features** using Scikit-learn's `OneHotEncoder` with `handle_unknown='ignore'`.
7. **Separate features and target** into `X` and `y`.
8. **Split the data** into training and testing sets with an 80:20 split and `random_state=42`.
9. **Train regression models**: Linear Regression, Ridge Regression, and Lasso Regression.
10. **Evaluate and compare** predictions using R² and actual-versus-predicted plots.

## Repository Structure

```text
house-price-prediction/
│
├── README.md
├── train.csv
└── Untitled7(1).ipynb
```

Suggested cleaner notebook filename:

```text
house_price_prediction.ipynb
```

## Requirements

Install the libraries used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Clone or download this project.
2. Put the dataset in the same directory as the notebook.
3. Rename the dataset to `train.csv` if necessary. The notebook currently loads it with:

   ```python
   data = pd.read_csv("train.csv")
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Untitled7(1).ipynb` (or your renamed notebook) and run the cells from top to bottom.

## Model Evaluation

The notebook calculates R² for each of the three regression models and creates a comparison table. It also plots actual versus predicted log prices.

R² measures how much of the variation in the target is explained by a model. A higher R² indicates a better fit on the evaluated test data, but it does not by itself establish that a model is ready for real-world deployment.

No fixed model score is stated here because the notebook file does not include saved outputs for the model evaluation cells. Run the notebook to generate the scores for your environment.

## Important Notes / Improvements

This README documents the workflow currently present in the supplied notebook. Before treating the model as production-ready, consider these improvements:

- **Prevent data leakage:** fit the encoder only on the training data. Prefer a Scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing is learned from the training split and consistently applied to test/new data.
- **Scale features for regularized models:** standardize numeric features before Ridge and Lasso, typically using a pipeline.
- **Use additional metrics:** report MAE and RMSE as well as R². If predictions are converted back from log space, evaluate prices in INR lakhs on the original scale.
- **Validate model performance:** use cross-validation and inspect residuals; tune Ridge/Lasso regularization parameters rather than relying only on defaults.
- **Validate input data:** check missing values, invalid values, outliers, and the meaning/coding of binary columns.
- **Make results reproducible:** save the preprocessing pipeline and trained model together, and record package versions and evaluation results.

## Current Scope

This is a notebook-based regression experiment. The supplied files do not currently include a web frontend, prediction API, saved trained model, automated tests, or deployment configuration.

## License

Add a license before distributing or reusing this project publicly.
