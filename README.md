# House Price Prediction

A tabular regression project using the included house-price CSV. The maintained entry point compares several regression models using a validation split, selects a candidate on validation data, refits it on train + validation, and evaluates the held-out test set once.

## Dataset

- File: `House Price Prediction Dataset.csv`
- Target: `Price`
- Features: area, bedrooms, bathrooms, floors, year built, location, condition and garage.
- The identifier `Id` is excluded from model features.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

## Evaluation

The pipeline imputes numeric and categorical data inside scikit-learn preprocessing, preventing preprocessing statistics from being learned from validation/test rows. Candidate selection uses validation R²; final metrics are calculated on the untouched test split. Results are written to `artifacts/metrics.json`, with the serialized model and test predictions alongside it.

The dataset is synthetic or externally sourced according to its provenance; verify its source and license before redistribution. Do not treat a score from one random split as a guarantee of real-world house values.

## Reproducibility

The random seed is fixed in `train.py`. Record Python and dependency versions, dataset provenance, and the generated metrics when publishing results. Generated artifacts are excluded from Git.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
