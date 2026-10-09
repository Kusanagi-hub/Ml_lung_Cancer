# Данные

Полный датасет `Lung Cancer Dataset.csv` (5000 строк, 18 колонок) в репозиторий
не коммитится (см. `.gitignore`: `data/`).

Схема колонок: `AGE GENDER SMOKING FINGER_DISCOLORATION MENTAL_STRESS
EXPOSURE_TO_POLLUTION LONG_TERM_ILLNESS ENERGY_LEVEL IMMUNE_WEAKNESS
BREATHING_ISSUE ALCOHOL_CONSUMPTION THROAT_DISCOMFORT OXYGEN_SATURATION
CHEST_TIGHTNESS FAMILY_HISTORY SMOKING_FAMILY_HISTORY STRESS_IMMUNE
PULMONARY_DISEASE (YES/NO)`.

Где взять: публичный набор "Lung Cancer Dataset" (Kaggle), 5000 строк.
Положите скачанный файл как `data/Lung Cancer Dataset.csv` и запустите:

```bash
bash run.sh --data "data/Lung Cancer Dataset.csv" --model rf --seed 42
```

Для воспроизводимости без скачивания `run.sh` сам создаёт
`data/sample.csv` (500 строк, та же схема, `src/make_sample.py`).
