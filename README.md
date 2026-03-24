# House Price Prediction

Predict house sale prices using the Ames Housing dataset (`train.csv`, `test.csv`) and generate a Kaggle submission file.

## Project Files

- `train.csv` - training data with `SalePrice`
- `test.csv` - test data without `SalePrice`
- `sample_submission.csv` - sample output format
- `house_pricing.ipynb` - notebook for training and prediction
- `submission.csv` - generated predictions for Kaggle upload

## Goal

Train a regression model and create `submission.csv` in this format:

- `Id`
- `SalePrice`

## Environment Setup

Use Python 3.9+ recommended.

Install required packages:

```bash
pip install numpy pandas scikit-learn
```

Optional (for higher Kaggle score):

```bash
pip install xgboost lightgbm catboost
```

## How to Run

1. Open `house_pricing.ipynb`
2. Run all cells from top to bottom
3. Confirm `submission.csv` is created
4. Upload `submission.csv` to Kaggle

## Baseline Pipeline (What It Does)

- Loads `train.csv` and `test.csv`
- Splits features and target (`SalePrice`)
- Handles missing values
- Encodes categorical columns
- Trains regression model
- Evaluates with RMSE
- Predicts test prices
- Saves `submission.csv`

## Improve Kaggle Score (Toward 0.20)

- Train on `log1p(SalePrice)` and convert back with `expm1()`
- Use boosting models (`XGBoost`, `LightGBM`, `CatBoost`)
- Blend multiple model predictions
- Use cross-validation for tuning

## Common Errors

### `TypeError: got an unexpected keyword argument 'squared'`

Some scikit-learn versions do not support:

```python
mean_squared_error(y_true, y_pred, squared=False)
```

Use:

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### `ModuleNotFoundError: No module named 'xgboost'`

Install missing package:

```bash
pip install xgboost
```

Or install all optional model packages:

```bash
pip install xgboost lightgbm catboost
```

## Submission Checklist

- `submission.csv` has exactly 2 columns: `Id`, `SalePrice`
- Number of rows matches `test.csv`
- No missing values in `SalePrice`
- File uploaded successfully to Kaggle

## Notes

- Start with a simple baseline, then improve step-by-step.
- Keep model random seeds fixed (`random_state=42`) for reproducibility.
