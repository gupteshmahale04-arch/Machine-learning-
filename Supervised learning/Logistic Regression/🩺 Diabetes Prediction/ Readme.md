
# 🩺 Diabetes Prediction Using Logistic Regression

An end-to-end Machine Learning classification project designed to predict the likelihood of diabetes in patients using physiological and demographic health indicators[cite: 70].

---

## 📌 Project Overview
This project develops a predictive pipeline utilizing binary classification to identify whether a patient has diabetes based on diagnostic measurements[cite: 70]. Due to the severe real-world consequences of false negatives in medical diagnostics, particular emphasis is placed on **Recall** to minimize missed diagnoses[cite: 70].

---

## 📊 Dataset Summary
- **Source:** Kaggle (`iammustafatz/diabetes-prediction-dataset`)[cite: 70]
- **Size:** 100,000 observations, 9 total columns[cite: 70]
- **Target Variable:** `diabetes` (Binary: `0` = No Diabetes, `1` = Diabetes)[cite: 70]
- **Class Distribution:** Highly imbalanced (~91,500 negative vs. ~8,500 positive cases)[cite: 70]

### Features[cite: 70]
| Feature | Type | Description[cite: 70] |
| :--- | :--- | :--- |
| `gender` | Categorical | Patient gender (`Female`, `Male`, `Other`)[cite: 70] |
| `age` | Numerical | Age in years[cite: 70] |
| `hypertension` | Binary | Blood pressure status (`0` = No, `1` = Yes)[cite: 70] |
| `heart_disease`| Binary | Heart disease history (`0` = No, `1` = Yes)[cite: 70] |
| `smoking_history` | Categorical | Smoking history categories (`never`, `current`, `No Info`, etc.)[cite: 70] |
| `bmi` | Numerical | Body Mass Index (weight in kg / height in m²)[cite: 70] |
| `HbA1c_level` | Numerical | Glycated hemoglobin test level[cite: 70] |
| `blood_glucose_level` | Numerical | Blood sugar concentration (mg/dL)[cite: 70] |

---

## 🛠️ Data Preprocessing Pipeline
1. **Exploratory Data Analysis (EDA):** Verified data distributions, missing values, and confirmed class imbalance[cite: 70].
2. **Categorical Encoding:** Applied `LabelEncoder` across object-type categorical variables (`gender`, `smoking_history`)[cite: 70].
3. **Train-Test Split:** Partitioned the data using an 80/20 train-test ratio (`test_size=0.2`, `random_state=42`)[cite: 70].
4. **Feature Scaling:** Applied `StandardScaler` to normalize feature ranges, preventing numerical bias in gradient descent and regularization penalties[cite: 70].

---

## 🤖 Modeling & Evaluation
- **Algorithm:** `LogisticRegression(class_weight="balanced", random_state=42)`[cite: 70]
- **Class Imbalance Strategy:** Incorporating `class_weight="balanced"` adjusts the loss function penalties inversely proportional to class frequencies, directly boosting sensitivity to diabetic cases[cite: 70].
- **Cross-Validation:** 10-fold cross-validation yielded a mean training recall score of **~87.88%**[cite: 70].

### Test Performance Metrics[cite: 70]
| Metric | Score[cite: 70] | Interpretation[cite: 70] |
| :--- | :---: | :--- |
| **Accuracy** | **0.89**[cite: 70] | Correct overall predictions across both classes[cite: 70]. |
| **Recall** | **0.88**[cite: 70] | Correctly detected ~88% of all true diabetic patients (primary medical focus)[cite: 70]. |
| **Precision** | **0.43**[cite: 70] | Trade-off due to balanced weighting, prioritizing recall over false alarms[cite: 70]. |
| **F1-Score** | **0.57**[cite: 70] | Harmonic balance between Precision and Recall[cite: 70]. |

---

## 🔮 Inference on Unseen Patient Data
The model supports live probability scoring for incoming patient records[cite: 70]:

```python
import pandas as pd

# Define input features for a new patient
new_patient = pd.DataFrame([{
    'gender': 1,               # Male
    'age': 50,
    'hypertension': 1,
    'heart_disease': 0,
    'smoking_history': 0,       # No Info
    'bmi': 30.5,
    'HbA1c_level': 7.0,
    'blood_glucose_level': 180
}])

# Transform features using the fitted scaler
new_patient_scaled = scaler.transform(new_patient)

# Generate class and probability predictions
prediction = clf.predict(new_patient_scaled)
probability = clf.predict_proba(new_patient_scaled)

print(f"No Diabetes : {probability[0][0]*100:.1f}%")
print(f"Has Diabetes : {probability[0][1]*100:.1f}%")
print(f"Prediction : {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

```

---

## 🚀 Future Enhancements



* Implement advanced ensemble algorithms such as **Random Forest** or **XGBoost**.


* Address class imbalance at the data level using **SMOTE** (Synthetic Minority Over-sampling Technique).


* Optimize classification probability thresholds to tune the precision-recall trade-off.
* Deploy an interactive clinical dashboard using **Streamlit**.


