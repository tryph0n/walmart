# Walmart Weekly Sales Prediction
> Regression project predicting Walmart weekly sales using Linear, Ridge, and Lasso regression with centralized preprocessing and chronological train/test splitting.

## Key Results

| Model | Test RMSE | Test R² | Overfitting |
|-------|-----------|---------|-------------|
| Linear Baseline | $579,969 | 0.3511 | 166.8% |
| Ridge (alpha=0.16) | $563,978 | 0.3864 | 157.8% |
| **Lasso (alpha=1000)** | **$555,210** | **0.4054** | **154.2%** |

Lasso is the best model, reducing RMSE by 4.3% and overfitting by 12.6 points versus baseline.

**Small dataset limitation:** 113 rows after cleaning, 21 features after encoding, yielding a 4:1 sample-to-feature ratio (recommended >= 10:1). Chronological split (70/30) respects temporal order but creates distribution shift between train and test sets.

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Data processing | pandas, numpy |
| ML | scikit-learn (LinearRegression, Ridge, Lasso, GridSearchCV, TimeSeriesSplit, StandardScaler) |
| Visualization | Plotly |
| Notebook | Jupyter |
| Package manager | uv |

## Installation

```bash
uv sync
```

## Usage

```bash
uv run jupyter lab
```

Execute notebooks in order:
1. `01_EDA.ipynb` -- Exploratory data analysis
2. `02-Baseline_Model.ipynb` -- Linear regression baseline
3. `03-Regularized_Regression.ipynb` -- Ridge and Lasso with GridSearchCV

## Data

- **Source:** `src/input/Walmart_Store_sales.csv` (151 rows, 8 columns)
- **Preprocessing:** `src/preprocessing.py` -- centralized pipeline (load, clean, group, encode, split)
- **Target:** Weekly_Sales
- **Split:** Chronological 70/30 with TimeSeriesSplit cross-validation (n_splits=3)
