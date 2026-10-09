#!/usr/bin/env bash
# Основной сценарий запуска ML-проекта: обучение + сохранение результатов.
# Использование:
#   bash run.sh [--model rf|logreg|knn|svm|dt] [--data PATH] [--seed N] [--epochs N] [--output-dir DIR]
# Пример:
#   bash run.sh --model rf --data data/sample.csv --seed 42 --epochs 100 --output-dir results
set -euo pipefail

MODEL="rf"
DATA="data/sample.csv"
SEED="42"
EPOCHS="100"
OUTPUT_DIR="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run.sh [--model rf|logreg|knn|svm|dt] [--data PATH] [--seed N] [--epochs N] [--output-dir DIR]";
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR" models data

# Если датасета нет — генерируем синтетический sample той же схемы (см. src/make_sample.py)
if [[ ! -f "$DATA" ]]; then
  echo "Data file $DATA not found, generating synthetic sample..."
  uv run python src/make_sample.py --output "$DATA" --seed "$SEED"
fi

echo "Training model=$MODEL data=$DATA seed=$SEED epochs=$EPOCHS -> $OUTPUT_DIR"
uv run python src/train.py \
  --model "$MODEL" \
  --data "$DATA" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --output-dir "$OUTPUT_DIR"

echo "Done. Results in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
