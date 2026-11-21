# 03_classification: класифікація підписки клієнтів на депозит

Цільова змінна: `Subscription` (0 = відмова, 1 = підписка). Датасет: UCI Bank Marketing.
Дані беруться з `../processed_data/` (`X_train_class.csv`, `X_test_class.csv`, `y_train_class.csv`, `y_test_class.csv`).

## Порядок запуску
```
python 01_tune_knn.py
python 02_tune_decision_tree.py
python 03_train_evaluate_models.py
python 04_visualizations.py
```

Скрипти можна запускати з будь-якої папки. Скрипт `04` потребує результатів `03`
(`results/params.json`, `metrics_comparison.csv`, `test_predictions.csv`), тому `03` потрібно запускати перед `04`.

## Що робить кожен скрипт

| Скрипт | Призначення |
|---|---|
| `01_tune_knn.py` | Підбір `k` і `weights` для k-NN (5-fold StratifiedKFold, метрика MCC); таблиця в терміналі та графік `images/knn_k_vs_mcc.png` |
| `02_tune_decision_tree.py` | Підбір `max_depth` і `criterion` для Decision Tree (GridSearchCV, MCC); графік якості та важливість ознак з фрагментом дерева (`images/`) |
| `03_train_evaluate_models.py` | Навчання 5 моделей (k-NN, Decision Tree, Logistic Regression, Random Forest, HistGradientBoosting), оцінка на тесті (Accuracy, Precision, Recall, F1, ROC-AUC, MCC); додатковий прогін без ознаки `Last Contact Duration` |
| `04_visualizations.py` | Порівняння моделей, матриці помилок, ROC/PR-криві, коефіцієнти та сигмоїда логістичної регресії, межі рішень (PCA) |

## Результати

- `results/metrics_comparison.csv`: метрики всіх моделей на тесті
- `results/metrics_without_duration.csv`: ті самі метрики без ознаки `Last Contact Duration`
- `results/test_predictions.csv`: передбачення та ймовірності на тестовій вибірці
- `results/params.json`: підібрані гіперпараметри (передаються зі скрипта `03` у `04`)
- `images/`: усі графіки (300 dpi)

## Примітки

- Гіперпараметри з `01` та `02` вручну вносяться в блок налаштувань скрипта `03`.
- `Last Contact Duration` залишено в основних моделях, але він відомий лише після дзвінка (витік даних),
  тому для оцінки впливу виконано окремий прогін без нього.
- Для k-NN та логістичної регресії використовується `StandardScaler` у `Pipeline`;
  для логістичної регресії видалено `Poutcome_unknown` (кореляція ≈ -0,9995 з `Previously_Contacted`).