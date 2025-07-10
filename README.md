# 🚜 Bulldozer Price Prediction

This project predicts the **resale price of used bulldozers** using machine learning. It’s based on the Kaggle competition [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers) and uses a **Random Forest Regressor** trained on over 400,000 rows of historical auction data.

---

## 📂 Dataset

- **Source**: Provided by Kaggle – Blue Book for Bulldozers
- **Target**: Predict the `SalePrice` of a bulldozer
- **Size**: ~401,000 rows, 50+ features
- **Files Used**: `TrainAndValid.csv`, `Test.csv`

---

## 🧠 Problem Statement

> Predict the sale price of a bulldozer given its characteristics (model, year made, product group, usage hours, etc.)

Goals:
- Handle missing values
- Encode categorical features
- Engineer meaningful date-based features
- Train a robust regression model
- Evaluate using RMSLE

---

## 📊 Model Used

- **RandomForestRegressor** (`scikit-learn`)
- Feature importance analysis
- Hyperparameter tuning with `RandomSearchCV`

---

## ⚙️ Key Steps

### ✅ Data Preprocessing
- Missing numerical: filled with median
- Missing categorical: filled with "Missing"
- Removed irrelevant or redundant features

### 🛠️ Feature Engineering
- Extracted date features from `saledate`
  - `saleYear`, `saleMonth`, `saleDayOfWeek`, etc.

---

## 🧪 Evaluation Metric and Results

We use **Root Mean Squared Log Error (RMSLE)** as the evaluation metric:

### 📐 RMSLE Formula:
```python
from sklearn.metrics import mean_squared_log_error
import numpy as np

def rmsle(y_true, y_preds):
    return np.sqrt(mean_squared_log_error(y_true, y_preds))
```

### Results: 

| Model Version            | Description                      | RMSLE Score |
| ------------------------ | -------------------------------- | ----------- |
| Baseline Random Forest   | No tuning, minimal preprocessing | 0.254       |
| With Feature Engineering | Added engineered date features   | 0.251       |
| Final Tuned Model        | Full pipeline + tuned parameters | **0.244**   |


## Loading the model:
```python
file=open("bulldozer_price_prediction_rf.pkl","rb")
model_loaded=pickle.load(file)
score_tmp(model_loaded)
```
