# California House Value Prediction

An educational machine-learning project that compares regression approaches for California housing values. The repository contains `House Price Prediction Dataset.csv` and a folder named `house price 0000`; inspect the folder to find the scripts or notebook and confirm the dataset columns before running anything.

## Setup

1. Clone or download this repository.
2. Review the CSV header, source, license, and target column.
3. Inspect `house price 0000` to identify the executable notebook/script and its expected working directory.
4. Create a Python virtual environment and install dependencies required by the source. No root `requirements.txt` is currently listed.
5. Follow the run instructions in the project code, if available.

The project description mentions Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, and KNN, with R², MAE, and RMSE for comparison. These are project-reported methods, not a benchmark reproduced by this README. Use a held-out test set and fit all preprocessing on training data only to avoid leakage.

## Tests and data handling

No root test command is documented. Avoid committing private or licensed data. Record dataset provenance, preprocessing choices, split strategy, and measured results so others can reproduce the experiment.


## Reproducible run

The training script is the extensionless root file `house price 0000`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
pip install -r requirements.txt
python "house price 0000"
```

**Dataset note:** the current script calls scikit-learn's `fetch_california_housing(as_frame=True)`; it does not read the separately checked-in `House Price Prediction Dataset.csv`. The first run may need internet access to download the California Housing dataset. The local CSV is not used by this script unless the code is deliberately changed.
