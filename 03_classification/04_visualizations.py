import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Підтягуємо збережені параметри моделей з попереднього скрипта
with open(os.path.join(RESULTS_DIR, "params.json"), "r", encoding="utf-8") as f:
    params = json.load(f)

BEST_K = params["BEST_K"]
BEST_WEIGHTS = params["BEST_WEIGHTS"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_CRITERION = params["TREE_CRITERION"]
LR_CLASS_WEIGHT = params["LR_CLASS_WEIGHT"]

# Завантажуємо метрики передбачення 
metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
pred = pd.read_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"))
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_class.csv"))["Subscription"]
model_names = list(metrics["Model"])

# 1 Порівняння моделей 
# Стовпчикова діаграма з усіма метриками на тесті
metrics.set_index("Model").plot(kind="bar", figsize=(12, 6), width=0.8)
plt.ylabel("Score")
plt.title("Model comparison on the test set")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.legend(loc="lower right", ncol=3)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "models_comparison.png"), dpi=300)
plt.close()

# Матриці помилок для всіх моделей
fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4.5))
for ax, name in zip(axes, model_names):
    cm = confusion_matrix(y_test, pred[name + "_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrices.png"), dpi=300)
plt.close()

# ROC та Precision-Recall криві
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
for name in model_names:
    proba = pred[name + "_proba"]

    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax_pr.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC curves")
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)

ax_pr.axhline(y_test.mean(), color="k", linestyle="--", label="Random")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall curves")
ax_pr.legend(loc="upper right")
ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "roc_pr_curves.png"), dpi=300)
plt.close()
print("Part 1 done: models_comparison, confusion_matrices, roc_pr_curves")

# 2 Важливість ознак та сигмоїда для логістичної регресії 
# Прибираємо Poutcome_unknown як базову категорію, щоб уникнути мультиколінеарності
X_train_lr = X_train.drop(columns=["Poutcome_unknown"])
X_test_lr = X_test.drop(columns=["Poutcome_unknown"])

lr = Pipeline([("scaler", StandardScaler()),
               ("lr", LogisticRegression(class_weight=LR_CLASS_WEIGHT, max_iter=1000, random_state=42))])
lr.fit(X_train_lr, y_train)

# Топ-15 найбільших за модулем коефіцієнтів
coefs = pd.Series(lr.named_steps["lr"].coef_[0], index=X_train_lr.columns)
top15 = coefs.reindex(coefs.abs().sort_values(ascending=False).head(15).index).iloc[::-1]

colors = ["tab:red" if c < 0 else "tab:green" for c in top15.values]
plt.figure(figsize=(9, 7))
plt.barh(top15.index, top15.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Логістична регресія: top-15 коефіцієнтів\n(зелений — збільшує шанс підписки, червоний — зменшує)")
plt.xlabel("Коефіцієнт (ознаки стандартизовані)")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "logistic_regression_insights.png"), dpi=300)
plt.close()

# Графік S-кривої (сигмоїди) на тестовій вибірці
z_scores = lr.decision_function(X_test_lr)
probs = lr.predict_proba(X_test_lr)[:, 1]
order = np.argsort(z_scores)

plt.figure(figsize=(9, 5))
plt.plot(z_scores[order], probs[order], color="tab:green", linewidth=2.5, label="Сигмоїда (ймовірність моделі)")
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Поріг рішення (0.5)")

# Беремо випадкові 400 точок, щоб не перевантажувати графік
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(y_test), size=400, replace=False)
for cls, color, lbl in [(0, "tab:red", "Відмова (0)"), (1, "tab:blue", "Підписка (1)")]:
    m = y_test.values[sample_idx] == cls
    plt.scatter(z_scores[sample_idx][m], y_test.values[sample_idx][m],
                c=color, alpha=0.5, s=25, label=lbl)

plt.title("Логістична регресія: S-крива ймовірності підписки на тестових даних")
plt.xlabel("Лінійна комбінація ознак (z)")
plt.ylabel("Ймовірність підписки P(Yes)")
plt.legend(loc="center right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "logistic_sigmoid_curve.png"), dpi=300)
plt.close()
print("Part 2 done: logistic_regression_insights, logistic_sigmoid_curve")

# 3 Межі рішень моделей (2D PCA)

# Знижуємо розмірність до 2 головних компонент, щоб намалювати площину
scaler = StandardScaler().fit(X_train)
pca = PCA(n_components=2, random_state=42).fit(scaler.transform(X_train))
Z_train = pca.transform(scaler.transform(X_train))
Z_test = pca.transform(scaler.transform(X_test))

models = {
    f"k-NN (k={BEST_K}, {BEST_WEIGHTS})": KNeighborsClassifier(n_neighbors=BEST_K, weights=BEST_WEIGHTS),
    f"Decision Tree ({TREE_CRITERION}, depth={TREE_MAX_DEPTH})": DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH, criterion=TREE_CRITERION, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight=LR_CLASS_WEIGHT, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

# Створюємо сітку координат для заливки областей класів
x_min, x_max = np.percentile(Z_train[:, 0], [1, 99])
y_min, y_max = np.percentile(Z_train[:, 1], [1, 99])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

rng = np.random.RandomState(42)
idx = rng.choice(len(Z_test), size=300, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.ravel()
for ax, (name, model) in zip(axes, models.items()):
    model.fit(Z_train, y_train)
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=["tab:blue", "tab:red"])
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        mask = y_test.values[idx] == cls
        ax.scatter(Z_test[idx][mask, 0], Z_test[idx][mask, 1], c=color, s=30,
                   edgecolor="k", linewidth=0.5, label=str(cls))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Межа рішень – {name}")
    ax.set_xlabel("Головна компонента 1 (PCA)")
    ax.set_ylabel("Головна компонента 2 (PCA)")
    ax.legend(title="Subscription", loc="upper right")

axes[-1].axis("off")  # ховаємо зайвий 6-й підграфік
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_boundaries.png"), dpi=300)
plt.close()
print("Part 3 done: decision_boundaries")

# 4 Підрахунок помилок (FN та FP)
rows = []
for name in model_names:
    tn, fp, fn, tp = confusion_matrix(y_test, pred[name + "_pred"]).ravel()
    rows.append({
        "Model": name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Missed subscribers (FN/(FN+TP))": fn / (fn + tp),  # скількох реальних підписників пропустили
        "False alarms (FP/(FP+TN))": fp / (fp + tn),        # скільки даремних дзвінків зробили
    })

errors = pd.DataFrame(rows).round(4)
errors.to_csv(os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False)
print("\nPart 4 done: error_analysis.csv")
print(errors.to_string(index=False))