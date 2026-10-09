"""Общие функции загрузки и предобработки данных.

Повторяет логику ноутбука ML_main_3.ipynb (ячейки 1-10):
- маппинг целевой колонки PULMONARY_DISEASE: YES/NO -> 1/0
- поиск выбросов по IQR и замена медианой для ENERGY_LEVEL/OXYGEN_SATURATION
- импутация категориальных пропусков + LabelEncoder
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

TARGET = "PULMONARY_DISEASE"
OUTLIER_COLS = ["ENERGY_LEVEL", "OXYGEN_SATURATION"]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"В файле {path} нет колонки {TARGET}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"NO": 0, "YES": 1})
    return df


def find_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> dict[str, list]:
    outliers: dict[str, list] = {}
    for col in cols:
        if col not in df.columns or df[col].isnull().all():
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bad = df[(df[col] < lower) | (df[col] > upper)]
        outliers[col] = bad.index.tolist()
    return outliers


def replace_outliers_with_median(
    df: pd.DataFrame, outliers_dict: dict[str, list]
) -> pd.DataFrame:
    df = df.copy()
    for col, idx in outliers_dict.items():
        if idx and col in df.columns:
            median_val = df[col].median()
            df.loc[idx, col] = median_val
    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Замена выбросов медианой только для вещественных колонок (как в ноутбуке)."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outliers = find_outliers_iqr(df, numeric_cols)
    filtered = {c: outliers[c] for c in OUTLIER_COLS if c in outliers}
    return replace_outliers_with_median(df, filtered)


def encode_categoricals(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Импутация + LabelEncoder для object-колонок. Возвращает X и fitted энкодеры."""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders: dict[str, LabelEncoder] = {}
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    return X, encoders


def apply_encoders(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    X = X.copy()
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(
                lambda v, _le=le: v if v in _le.classes_ else _le.classes_[0]
            )
            X[col] = le.transform(X[col])
    return X
