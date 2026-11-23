import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ==============================================================================
# 1. Шляхи та параметри
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
DATA_DIR = ROOT / "processed_data"
IMAGES_DIR = CURRENT_DIR / "images"
RESULTS_DIR = CURRENT_DIR / "results"
IMAGES_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "params.json", "r", encoding="utf-8") as f:
    params = json.load(f)

BEST_K = params["BEST_K"]
BEST_WEIGHTS = params["BEST_WEIGHTS"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_CRITERION = params["TREE_CRITERION"]
LR_CLASS_WEIGHT = params["LR_CLASS_WEIGHT"]

# ==============================================================================
# 2. Завантаження даних
# ==============================================================================
X_train = pd.read_csv(DATA_DIR / "X_train_class.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_class.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_class.csv")["Subscription"].astype(int)
y_test = pd.read_csv(DATA_DIR / "y_test_class.csv")["Subscription"].astype(int)

features_7 = [
    "Age", "Balance (euros)", "Education", "Housing Loan_yes",
    "Personal Loan_yes", "Marital Status_married", "Marital Status_single",
]
cols_no_dur = [c for c in X_train.columns if c != "Last Contact Duration"]

def get_models():
    return {
        f"k-NN (k={BEST_K}, {BEST_WEIGHTS})": KNeighborsClassifier(n_neighbors=BEST_K, weights=BEST_WEIGHTS),
        f"Decision Tree ({TREE_CRITERION}, depth={TREE_MAX_DEPTH})": DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH, criterion=TREE_CRITERION, random_state=42),
        "Logistic Regression (balanced)": LogisticRegression(
            class_weight=LR_CLASS_WEIGHT, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight=LR_CLASS_WEIGHT),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42,  class_weight=LR_CLASS_WEIGHT)
    }

# ==============================================================================
# ГРАФІК 1: Порівняння кластеризації та класифікації (7 спільних ознак)
# ==============================================================================
X_all_7 = pd.concat([X_train[features_7], X_test[features_7]], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True).to_numpy()

limit_all = X_all_7["Balance (euros)"].quantile(0.95)
X_all_7["Balance (euros)"] = X_all_7["Balance (euros)"].clip(upper=limit_all)

pca_all = PCA(n_components=2, random_state=42)
Z_all_7 = pca_all.fit_transform(X_all_7)
for i in range(2):
    if pca_all.components_[i].sum() < 0:
        pca_all.components_[i] *= -1
        Z_all_7[:, i] *= -1

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_all = kmeans.fit_predict(X_all_7)
centers_2d = pca_all.transform(kmeans.cluster_centers_)

classifier = HistGradientBoostingClassifier(random_state=42)
classifier.fit(X_train[cols_no_dur], y_train)
y_pred_test = classifier.predict(X_test[cols_no_dur])
Z_test_all = Z_all_7[len(X_train):]

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_all_7), size=min(6000, len(X_all_7)), replace=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
class_colors = {0: "#1f77b4", 1: "#d62728"}

for c in range(4):
    idx = sample_idx[clusters_all[sample_idx] == c]
    axes[0].scatter(Z_all_7[idx, 0], Z_all_7[idx, 1], s=8, alpha=0.5,
                    color=cluster_colors[c], label=f"Кластер {c}")
axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], s=180, marker="X", color="black", label="Центри")
axes[0].set_title("K-Means: 4 кластери")

for c, label in [(0, "Не відкрили"), (1, "Відкрили")]:
    idx = sample_idx[y_all[sample_idx] == c]
    axes[1].scatter(Z_all_7[idx, 0], Z_all_7[idx, 1], s=8, alpha=0.5,
                    color=class_colors[c], label=label)
axes[1].set_title("Фактичне відкриття депозиту")

for c, label in [(0, "Прогноз: ні"), (1, "Прогноз: так")]:
    mask = y_pred_test == c
    axes[2].scatter(Z_test_all[mask, 0], Z_test_all[mask, 1], s=8, alpha=0.5,
                    color=class_colors[c], label=label)
axes[2].set_title(f"Прогноз на тестовій вибірці ({len(cols_no_dur)} ознак)")

pc1_a, pc2_a = 100 * pca_all.explained_variance_ratio_
for ax in axes:
    ax.set_xlabel(f"Головна компонента 1 ({pc1_a:.1f}%)")
    ax.set_ylabel(f"Головна компонента 2 ({pc2_a:.1f}%)")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        if hasattr(lh, "set_sizes"):
            lh.set_sizes([35])

plt.suptitle("Порівняння кластеризації та класифікації через PCA (7 спільних ознак)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "pca_clusters_vs_classification.png", dpi=200)
plt.close()

# ==============================================================================
# ГРАФІК 2: Межі рішень у просторі 7 ознак (як у кластеризації)
# ==============================================================================
X_tr_7 = X_train[features_7].copy()
X_te_7 = X_test[features_7].copy()
bal_lim = X_tr_7["Balance (euros)"].quantile(0.95)
X_tr_7["Balance (euros)"] = X_tr_7["Balance (euros)"].clip(upper=bal_lim)
X_te_7["Balance (euros)"] = X_te_7["Balance (euros)"].clip(upper=bal_lim)

pca_7 = PCA(n_components=2, random_state=42)
Z_tr_7 = pca_7.fit_transform(X_tr_7)
Z_te_7 = pca_7.transform(X_te_7)
for i in range(2):
    if pca_7.components_[i].sum() < 0:
        pca_7.components_[i] *= -1
        Z_tr_7[:, i] *= -1
        Z_te_7[:, i] *= -1

pc1_7, pc2_7 = 100 * pca_7.explained_variance_ratio_
xx, yy = np.meshgrid(np.linspace(-3.3, 4.0, 250), np.linspace(-3.8, 2.1, 250))
grid_7 = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (name, model) in zip(axes, get_models().items()):
    model.fit(Z_tr_7, y_train)
    zz = model.predict(grid_7).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.28, levels=[-0.5, 0.5, 1.5], colors=["#1f77b4", "#d62728"])
    for cls, color, lbl, alpha, size in [(0, "#1f77b4", "Ні (0)", 0.30, 5), (1, "#d62728", "Так (1)", 0.45, 6)]:
        mask = y_test.values == cls
        ax.scatter(Z_te_7[mask, 0], Z_te_7[mask, 1], c=color, s=size, alpha=alpha, linewidth=0, label=lbl)
    ax.set_xlim(-3.3, 4.0)
    ax.set_ylim(-3.8, 2.1)
    ax.set_title(f"Межа рішень: {name}")
    ax.set_xlabel(f"PC1 ({pc1_7:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_7:.1f}%)")
    ax.grid(alpha=0.25)
    leg = ax.legend(title="Підписка (test)", loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([30])

axes[-1].axis("off")
fig.suptitle(f"Межі рішень у просторі 7 ознак кластеризації (test: {len(Z_te_7)} клієнтів)", fontsize=14)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "decision_boundaries_7features.png", dpi=200)
plt.close()

# ==============================================================================
# ГРАФІК 3: Межі рішень на всіх ознаках до дзвінка (окремий PCA для класифікації)
# ==============================================================================
scaler_full = StandardScaler()
Z_tr_full = PCA(n_components=2, random_state=42).fit(scaler_full.fit_transform(X_train[cols_no_dur]))
pca_full = PCA(n_components=2, random_state=42)
Z_tr_full = pca_full.fit_transform(scaler_full.transform(X_train[cols_no_dur]))
Z_te_full = pca_full.transform(scaler_full.transform(X_test[cols_no_dur]))

pc1_f, pc2_f = 100 * pca_full.explained_variance_ratio_
x_min, x_max = np.percentile(Z_tr_full[:, 0], [1, 99])
y_min, y_max = np.percentile(Z_tr_full[:, 1], [1, 99])
xx_f, yy_f = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_f = np.c_[xx_f.ravel(), yy_f.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (name, model) in zip(axes, get_models().items()):
    model.fit(Z_tr_full, y_train)
    zz = model.predict(grid_f).reshape(xx_f.shape)
    ax.contourf(xx_f, yy_f, zz, alpha=0.28, levels=[-0.5, 0.5, 1.5], colors=["#1f77b4", "#d62728"])
    for cls, color, lbl, alpha, size in [(0, "#1f77b4", "Ні (0)", 0.25, 5), (1, "#d62728", "Так (1)", 0.55, 7)]:
        mask = y_test.values == cls
        ax.scatter(Z_te_full[mask, 0], Z_te_full[mask, 1], c=color, s=size, alpha=alpha, linewidth=0, label=lbl)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Межа рішень: {name}")
    ax.set_xlabel(f"PC1 ({pc1_f:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_f:.1f}%)")
    ax.grid(alpha=0.25)
    leg = ax.legend(title="Підписка (test)", loc="upper right")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([30])

axes[-1].axis("off")
fig.suptitle(f"Межі рішень класифікаторів на всіх {len(cols_no_dur)} ознаках до дзвінка (test: {len(Z_te_full)} клієнтів)", fontsize=14)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "decision_boundaries_all_features.png", dpi=200)
plt.close()

print("Готово! Збережено всі 3 графіки у:", IMAGES_DIR)