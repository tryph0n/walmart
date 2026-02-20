"""Walmart Sales Prediction - Shared preprocessing module.

Centralizes data loading, cleaning, feature engineering, and train-test
splitting logic used across all project notebooks.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(csv_path: str = "src/input/Walmart_Store_sales.csv") -> pd.DataFrame:
    """Load CSV and create date-derived features.

    Returns the raw DataFrame with added Year, Month, Day, DayOfWeek columns.
    Store is cast to int and Holiday_Flag to int (after NaN handling by caller).
    """
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Store is always present (no NaN in original data) -> int
    df["Store"] = df["Store"].astype(int)

    # Holiday_Flag -> int only where not NaN (keep NaN as-is for now;
    # clean_data will handle them depending on strategy)
    # We use pd.Int64Dtype to allow nullable int
    df["Holiday_Flag"] = df["Holiday_Flag"].astype(pd.Int64Dtype())

    return df


def clean_data(df: pd.DataFrame, strategy: str = "impute") -> pd.DataFrame:
    """Clean the DataFrame according to the chosen strategy.

    Parameters
    ----------
    df : DataFrame returned by load_raw_data.
    strategy : 'drop' (notebook legacy) or 'impute' (default).

    Returns
    -------
    Cleaned DataFrame with outliers removed.
    """
    numeric_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    df = df.copy()

    if strategy == "drop":
        # Legacy notebook behaviour: drop ALL rows with any NaN
        df = df.dropna(subset=["Weekly_Sales"]).copy()
        df = df.dropna(subset=["Holiday_Flag"]).copy()
        df = df.dropna(subset=["Date"]).copy()
        df = df.dropna(subset=numeric_cols).copy()
    elif strategy == "impute":
        # Drop only rows where target or date is NaN (non-negotiable)
        df = df.dropna(subset=["Weekly_Sales"]).copy()
        df = df.dropna(subset=["Date"]).copy()

        df["Holiday_Flag"] = df["Holiday_Flag"].fillna(0).astype(int)

        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'drop' or 'impute'.")

    # Bounds computed ONCE on the current dataset
    outliers_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        outliers_mask |= (df[col] < lower) | (df[col] > upper)

    df = df[~outliers_mask].copy()

    df["Holiday_Flag"] = df["Holiday_Flag"].astype(int)

    return df


def group_stores(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    """Add Store_Grouped column; rare stores (< min_samples) mapped to 999."""
    df = df.copy()
    store_counts = df["Store"].value_counts()
    rare_stores = store_counts[store_counts < min_samples].index.tolist()

    df["Store_Grouped"] = df["Store"].copy()
    if rare_stores:
        df.loc[df["Store"].isin(rare_stores), "Store_Grouped"] = 999

    return df


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_features, numeric_features) for the preprocessor.

    DayOfWeek is excluded if it has zero variance (e.g. all Fridays).
    """
    categorical_features = ["Store_Grouped", "Holiday_Flag"]

    numeric_features = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Day",
    ]

    if "DayOfWeek" in df.columns and df["DayOfWeek"].std() > 0:
        numeric_features.append("DayOfWeek")

    return categorical_features, numeric_features


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer with OneHotEncoder + StandardScaler."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )


def split_chronological(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Chronological train/test split (no shuffle).

    Returns X_train_raw, X_test_raw, y_train, y_test, split_idx.
    """
    sorted_idx = df["Date"].argsort()
    X = X.iloc[sorted_idx].reset_index(drop=True)
    y = y.iloc[sorted_idx].reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    X_train_raw = X.iloc[:split_idx]
    X_test_raw = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train_raw, X_test_raw, y_train, y_test, split_idx


def prepare_data(
    csv_path: str = "src/input/Walmart_Store_sales.csv",
    strategy: str = "drop",
    test_size: float = 0.3,
) -> dict:
    """Full pipeline: load -> clean -> group -> split -> preprocess.

    Returns a dict with all artifacts needed by notebooks.
    """
    df_raw = load_raw_data(csv_path)

    df_clean = clean_data(df_raw, strategy=strategy)

    df_clean = group_stores(df_clean)

    df_clean = df_clean.sort_values("Date").reset_index(drop=True)

    drop_cols = ["Weekly_Sales", "Date", "Store"]
    X = df_clean.drop(columns=drop_cols)
    y = df_clean["Weekly_Sales"]

    categorical_features, numeric_features = get_feature_lists(df_clean)

    X_train_raw, X_test_raw, y_train, y_test, split_idx = split_chronological(
        df_clean, X, y, test_size=test_size
    )

    # Fit on train only to prevent data leakage
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }
