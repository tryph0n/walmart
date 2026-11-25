# Walmart Weekly Sales Prediction
> Regression project predicting Walmart weekly sales using Linear, Ridge, and Lasso regression with centralized preprocessing and chronological train/test splitting.

Fictional business case from a training course. No affiliation with Walmart Inc.

## Key Results

Test-set results from the committed outputs of `03-Regularized_Regression.ipynb`:

| Model | Test RMSE | Test R² | Overfitting (RMSE gap) |
|-------|-----------|---------|------------------------|
| Linear Baseline | $1,026,115 | -1.6130 | 143.7% |
| **Ridge (alpha=3.56)** | **$820,985** | **-0.6727** | **72.7%** |
| Lasso (alpha=1000) | $994,303 | -1.4535 | 135.9% |

Ridge is the best of the three models: test RMSE 20.0% lower than the baseline and an overfitting gap of 72.7% instead of 143.7%. All test R² values are negative: on this data, the linear models predict worse than the mean of the test set. Lasso selects the upper bound of its alpha grid and removes 0 of 13 features. Next step: non-linear models (for example tree-based models).

**Small dataset limitation:** 71 rows after cleaning (complete cases only), 13 features after encoding, 49 train / 22 test rows, a 3.8:1 sample-to-feature ratio (recommended >= 10:1). Chronological split (70/30) respects temporal order but creates distribution shift between train and test sets.

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13 |
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

PNG export (`fig.write_image`) uses kaleido 1.x, which needs Chrome. Run `uv run plotly_get_chrome` once before re-running the notebooks.

## Data

- **Source:** `src/input/Walmart_Store_sales.csv` (150 rows, 8 columns)
- **Provenance:** course dataset of the Jedha data science training, adapted by the course from a Kaggle competition dataset (the course brief does not name the competition). No licence stated by the source; the data is not covered by the code licence.
- **Preprocessing:** `src/preprocessing.py` -- centralized pipeline (load, clean, group, encode, split)
- **Target:** Weekly_Sales
- **Split:** Chronological 70/30 with TimeSeriesSplit cross-validation (n_splits=3)
- **Figures:** `reports/figures/` (PNG exports written by the notebooks)

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Notebooks are excluded from ruff so they keep the code that produced their outputs.

## License

MIT, see [LICENSE](LICENSE). The data is not covered by this licence.
