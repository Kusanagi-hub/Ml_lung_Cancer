# Данные

Основной датасет — `data/Lung Cancer Dataset.csv` (5000 строк, 18 колонок,
в репозиторий не коммитится, см. `.gitignore`: `data/*`).

Схема колонок: `AGE GENDER SMOKING FINGER_DISCOLORATION MENTAL_STRESS
EXPOSURE_TO_POLLUTION LONG_TERM_ILLNESS ENERGY_LEVEL IMMUNE_WEAKNESS
BREATHING_ISSUE ALCOHOL_CONSUMPTION THROAT_DISCOMFORT OXYGEN_SATURATION
CHEST_TIGHTNESS FAMILY_HISTORY SMOKING_FAMILY_HISTORY STRESS_IMMUNE
PULMONARY_DISEASE (YES/NO)`. Пропусков нет, классы: NO 2963 / YES 2037.

Запуск на полных данных:

```bash
bash run.sh --model rf --seed 42 --epochs 100 --output-dir results
```

Если файла нет (например, на чужой машине после `git clone`), `run.sh` сам
откатится на `data/sample.csv` — синтетику той же схемы на 500 строк
(`src/make_sample.py`).
