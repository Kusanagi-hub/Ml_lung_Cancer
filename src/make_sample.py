"""Генератор синтетического sample-датасета той же схемы, что и Lung Cancer Dataset.

Используется только чтобы `bash run.sh` работал из коробки без скачивания
полного датасета (5000 строк). Распределения приближены к describe() из ноутбука:
AGE 30-84, бинарные признаки 0/1, ENERGY_LEVEL ~ N(55, 8), OXYGEN_SATURATION ~ N(95, 2).
Целевой признак коррелирует с SMOKING/BREATHING_ISSUE/AGE.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

COLUMNS = [
    "AGE",
    "GENDER",
    "SMOKING",
    "FINGER_DISCOLORATION",
    "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS",
    "ENERGY_LEVEL",
    "IMMUNE_WEAKNESS",
    "BREATHING_ISSUE",
    "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT",
    "OXYGEN_SATURATION",
    "CHEST_TIGHTNESS",
    "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE",
    "PULMONARY_DISEASE",
]


def make_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["AGE"] = rng.integers(30, 85, n)
    df["GENDER"] = rng.binomial(1, 0.50, n)
    df["SMOKING"] = rng.binomial(1, 0.67, n)
    df["FINGER_DISCOLORATION"] = rng.binomial(1, 0.60, n)
    df["MENTAL_STRESS"] = rng.binomial(1, 0.55, n)
    df["EXPOSURE_TO_POLLUTION"] = rng.binomial(1, 0.55, n)
    df["LONG_TERM_ILLNESS"] = rng.binomial(1, 0.45, n)
    df["ENERGY_LEVEL"] = np.round(rng.normal(55.0, 8.0, n), 2)
    df["IMMUNE_WEAKNESS"] = rng.binomial(1, 0.50, n)
    df["BREATHING_ISSUE"] = rng.binomial(1, 0.50, n)
    df["ALCOHOL_CONSUMPTION"] = rng.binomial(1, 0.50, n)
    df["THROAT_DISCOMFORT"] = rng.binomial(1, 0.50, n)
    df["OXYGEN_SATURATION"] = np.round(rng.normal(95.0, 2.0, n), 2)
    df["CHEST_TIGHTNESS"] = rng.binomial(1, 0.45, n)
    df["FAMILY_HISTORY"] = rng.binomial(1, 0.40, n)
    df["SMOKING_FAMILY_HISTORY"] = rng.binomial(1, 0.50, n)
    df["STRESS_IMMUNE"] = rng.binomial(1, 0.50, n)
    # Скор риска -> вероятность YES (дисбаланс ~60/40 как в оригинале)
    score = (
        0.9 * df["SMOKING"]
        + 0.7 * df["BREATHING_ISSUE"]
        + 0.4 * df["FINGER_DISCOLORATION"]
        + 0.02 * (df["AGE"] - 55)
        - 0.05 * (df["ENERGY_LEVEL"] - 55)
        - 0.3 * (df["OXYGEN_SATURATION"] - 95)
        + rng.normal(0, 0.8, n)
    )
    proba = 1 / (1 + np.exp(-score))
    df["PULMONARY_DISEASE"] = np.where(rng.random(n) < proba, "YES", "NO")
    return df[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/sample.csv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = make_sample(a.n, a.seed)
    df.to_csv(a.output, index=False)
    print(f"saved {a.output}: shape={df.shape}")


if __name__ == "__main__":
    main()
