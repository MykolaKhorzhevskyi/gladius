#  k-NN та Decision Tree гіперпараметри підібрані 
#  Logistic Regression використано class_weight='balanced' через дисбаланс класів 88/12;
#  Random Forest (n_estimators=200) та HistGradientBoosting: базові параметри за замовчуванням.

import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_class.csv"))["Subscription"]

print("Розподіл класів у train (0 = Ні, 1 = Так):")
print(y_train.value_counts(normalize=True).round(3).to_string())
print("-" * 50)

# Найкращі параметри з попередніх скриптів
BEST_K = 7
BEST_WEIGHTS = "distance"
TREE_MAX_DEPTH = 12
TREE_CRITERION = "entropy"
LR_CLASS_WEIGHT = "balanced"

print(f"KNN: k={BEST_K}, weights={BEST_WEIGHTS} | Tree: {TREE_MAX_DEPTH}, {TREE_CRITERION} | LR: {LR_CLASS_WEIGHT}")

# Для логістичної регресії прибираємо Poutcome_unknown, бо вона дублює Previously_Contacted (мультиколінеарність)
lr_drop = ["Poutcome_unknown"]


models = {
    "k-NN": (Pipeline([("scaler", StandardScaler()),
                       ("knn", KNeighborsClassifier(n_neighbors=BEST_K, weights=BEST_WEIGHTS))]), []),
    "Decision Tree": (DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, criterion=TREE_CRITERION,
                                             random_state=42), []),
    "Logistic Regression": (Pipeline([("scaler", StandardScaler()),
                                      ("lr", LogisticRegression(class_weight=LR_CLASS_WEIGHT,
                                                                max_iter=1000, random_state=42))]),
                            lr_drop),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), []),
    "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=42), []),
}

metrics_rows = []
predictions = pd.DataFrame({"y_true": y_test})

for name, (model, drop_cols) in models.items():
    Xtr = X_train.drop(columns=drop_cols)
    Xte = X_test.drop(columns=drop_cols)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    predictions[name + "_pred"] = y_pred
    predictions[name + "_proba"] = y_proba
    print("Done:", name)

metrics = pd.DataFrame(metrics_rows).round(4)
metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
predictions.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "params.json"), "w", encoding="utf-8") as f:
    json.dump({
        "BEST_K": BEST_K,
        "BEST_WEIGHTS": BEST_WEIGHTS,
        "TREE_MAX_DEPTH": TREE_MAX_DEPTH,
        "TREE_CRITERION": TREE_CRITERION,
        "LR_CLASS_WEIGHT": LR_CLASS_WEIGHT,
    }, f, indent=2)

print("\n" + metrics.to_string(index=False))
