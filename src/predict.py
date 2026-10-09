"""Инференс сохранённой модели на новом CSV."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import TARGET, apply_encoders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Путь к .pkl из models/")
    ap.add_argument("--input", required=True, help="CSV без целевой колонки (или с ней)")
    ap.add_argument("--output", default="results/predictions.csv")
    a = ap.parse_args()

    with open(a.model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    encoders = bundle.get("encoders", {}) if isinstance(bundle, dict) else {}
    feature_columns = bundle.get("feature_columns") if isinstance(bundle, dict) else None
    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None

    df = pd.read_csv(a.input)
    if TARGET in df.columns:
        df = df.drop(TARGET, axis=1)
    if feature_columns:
        df = df.reindex(columns=feature_columns)
    df = apply_encoders(df, encoders)
    X = scaler.transform(df) if scaler is not None else df
    preds = model.predict(X)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    print(f"saved {out}: n={len(preds)}")


if __name__ == "__main__":
    main()
