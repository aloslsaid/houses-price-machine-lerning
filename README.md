# California Housing Value Prediction

> A comparative regression experiment using California housing features to estimate median property values.

## Overview
This project builds and compares classical machine-learning regressors on the California Housing dataset. It focuses on a clear supervised-learning workflow: selecting the target, splitting the data, scaling where appropriate, comparing algorithms, inspecting feature importance, and evaluating predictions.

## Dataset and target
The training script loads the California Housing dataset through Scikit-learn's `fetch_california_housing` interface. The target is `MedHouseVal`; the remaining columns are used as predictors. The script also includes a CSV file in the repository, but the current training script obtains its active dataset through Scikit-learn rather than reading that CSV.

## Algorithms compared
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

The script scales features for Linear Regression, Ridge, and KNN, while tree-based models use the unscaled input features.

## Evaluation and outputs
Models are compared using:
- **R²** — proportion of target variance explained by the model.
- **MAE** — average absolute prediction error.
- **RMSE** — error measure that penalizes larger misses more strongly.

The highest-R² candidate is selected and the script prints its test metrics. For supported tree models, it also reports feature importance and demonstrates a prediction for one held-out example.

## Installation
Python 3.10+ is recommended.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn
```

## Run
The Python script is named `house price 0000` and has no file extension:
```bash
python "house price 0000"
```
The first run may need an internet connection because Scikit-learn may download the California Housing dataset if it is not already cached locally.

## Repository contents
- `house price 0000` — model training and comparison script.
- `House Price Prediction Dataset.csv` — accompanying CSV dataset.

## Scope and limitations
This is a machine-learning demonstration, not a property appraisal system. Dataset values have a defined geographic and historical context; predictions should not be interpreted as current listing prices. Performance on a held-out split does not guarantee performance in another region or market period.

## Skills demonstrated
Regression · Dataset exploration · Feature scaling · Ensemble learning · Model comparison · R² / MAE / RMSE · Feature importance
