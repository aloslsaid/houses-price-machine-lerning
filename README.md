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


## Evaluation integrity

The training script uses separate training, validation, and test partitions. Model selection is based on validation R²; the held-out test split is used only for the selected model's final metrics. The chosen estimator is then refit on the development data for the example prediction. The script fetches the California Housing dataset through scikit-learn; it does not currently consume the separately stored CSV file.
