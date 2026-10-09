"""Обучение модели детекции лёгочных заболеваний.

Соответствует ячейкам 10-37 ноутбука ML_main_3.ipynb:
split (stratify) -> SMOTE -> обучение -> метрики.
Основная модель: RandomForest. Остальные выбираются параметром --model.

Артефакты в --output-dir (results/):
  metrics.json, classification_report.txt, confusion_matrix.png,
  feature_importance.csv (для rf/dt)
Модель: --models-dir/<model>_seed<seed>.pkl (dict: model, encoders, feature_columns)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils import (
    TARGET,
    apply_encoders,
    clean_outliers,
    encode_categoricals,
    encode_target,
    load_dataset,
)


def build_model(name: str, seed: int, epochs: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=epochs, random_state=seed)
    if name == "logreg":
        return LogisticRegression(random_state=seed, max_iter=max(epochs, 200))
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "svm":
        return SVC(random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(max_depth=5, random_state=seed)
    raise ValueError(f"Неизвестная модель: {name} (rf/logreg/knn/svm/dt)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение модели Lung Disease Detection")
    ap.add_argument("--data", required=True, help="Путь к CSV-датасету")
    ap.add_argument("--model", default="rf", choices=["rf", "logreg", "knn", "svm", "dt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument(
        "--epochs", type=int, default=100, help="n_estimators для RF / max_iter для logreg"
    )
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument(
        "--tune",
        action="store_true",
        help="GridSearchCV для RF (сетка из ячейки 35 ноутбука, cv=3)",
    )
    a = ap.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)

    df = encode_target(clean_outliers(load_dataset(a.data)))
    X = df.drop(TARGET, axis=1)
    y = df[TARGET].dropna()
    X = X.loc[y.index]
    X, encoders = encode_categoricals(X)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, random_state=a.seed, stratify=y
    )
    if not a.no_smote:
        X_train, y_train = SMOTE(random_state=a.seed).fit_resample(X_train, y_train)

    model = build_model(a.model, a.seed, a.epochs)
    if a.model == "svm":  # как в ячейке 17 ноутбука: StandardScaler для SVM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    _ = apply_encoders  # predict.py использует те же энкодеры из pkl
    best_params = None
    if a.tune:
        if a.model != "rf":
            raise ValueError("--tune поддерживается только для --model rf")
        grid = GridSearchCV(
            RandomForestClassifier(random_state=a.seed),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
        print(f"best_params={best_params}")
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    metrics = {
        "model": a.model,
        "seed": a.seed,
        "epochs": a.epochs,
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "tuned": bool(a.tune),
        "best_params": best_params,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / "classification_report.txt").write_text(
        f"model={a.model} seed={a.seed}\n\n"
        + classification_report(y_test, y_pred, zero_division=0)
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix ({a.model}, acc={acc:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            out / "feature_importance.csv", index=False
        )

    model_path = Path(a.models_dir) / f"{a.model}_seed{a.seed}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "encoders": encoders,
                "feature_columns": feature_columns,
            },
            f,
        )
    print(f"model={a.model} accuracy={acc:.4f}")
    print(f"saved: {out}/metrics.json, confusion_matrix.png, {model_path}")


if __name__ == "__main__":
    main()
