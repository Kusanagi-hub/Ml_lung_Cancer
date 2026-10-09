# Lung Disease Detection — организация инженерного окружения ML-проекта

Классификация лёгочных заболеваний по таблице симптомов (`Lung Cancer Dataset`,
5000 строк, 18 колонок). Код вынесен из `ML_main_3.ipynb` в воспроизводимый
пайплайн: подготовка данных → обучение → предсказания.

Основная модель — **RandomForest** (`--model rf`). Для сравнения доступны
`logreg / knn / svm / dt` (см. ячейки 13–29 ноутбука).

## Структура

```
src/train.py        обучение + метрики + сохранение модели
src/predict.py      инференс сохранённой модели
src/utils.py        загрузка, чистка выбросов (IQR+медиана), кодирование
src/make_sample.py  генератор синтетического sample-датасета той же схемы
configs/config.yaml значения по умолчанию
data/               датасеты (в git не коммитятся, кроме README)
models/             сохранённые .pkl (в git не коммитятся)
results/            метрики каждого запуска (metrics.json, confusion_matrix.png, ...)
run.sh              точка входа: обучение одной командой
```

Django-части (`HospitalSystem/`, `manage.py`, `main/`, `dbwork/`) к заданию не
относятся и в репозиторий не коммитятся.

## Требования

Linux, Python >= 3.10, [uv](https://astral.sh/uv/).

## Установка

```bash
uv sync
```

Зависимости (`pyproject.toml`): `numpy pandas scikit-learn matplotlib seaborn
imbalanced-learn pyyaml`. Проверка с нуля: `rm -rf .venv && uv sync`.

## Запуск

```bash
bash run.sh --model rf --data data/sample.csv --seed 42 --epochs 100 --output-dir results
```

Если `--data` отсутствует, скрипт сам сгенерирует `data/sample.csv`
(`src/make_sample.py`, 500 строк, та же схема). Полный датасет положите как
`data/Lung Cancer Dataset.csv` (см. `data/README.md`) и передайте его в `--data`.

Предсказания на новых данных:

```bash
uv run python src/predict.py --model-path models/rf_seed42.pkl \
  --input data/sample.csv --output results/predictions.csv
```

## Bash-скрипт `run.sh`

Параметры (все необязательные, есть значения по умолчанию):

| Параметр | Описание |
|---|---|
| `--model` | `rf` (default), `logreg`, `knn`, `svm`, `dt` |
| `--data` | путь к CSV (default `data/sample.csv`) |
| `--seed` | random seed (default `42`) |
| `--epochs` | `n_estimators` для RF / `max_iter` для LogReg (default `100`) |
| `--output-dir` | каталог результатов (default `results`) |

Скрипт создаёт каталоги (`mkdir -p`), при необходимости генерирует данные,
запускает `src/train.py` и складывает в `--output-dir`: `metrics.json`,
`classification_report.txt`, `confusion_matrix.png`, `feature_importance.csv`
(для `rf/dt`). Модель сохраняется в `models/<model>_seed<seed>.pkl`.
`bash run.sh --help` — краткая справка.
