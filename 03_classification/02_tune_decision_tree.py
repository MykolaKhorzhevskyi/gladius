import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI window
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, GridSearchCV

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "max_depth": [3, 5, 8, 12, None],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                    scoring="matthews_corrcoef", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

table = pd.DataFrame({
    "criterion": grid.cv_results_["param_criterion"],
    "max_depth": [str(d) for d in grid.cv_results_["param_max_depth"]],
    "MCC (mean)": grid.cv_results_["mean_test_score"],
    "MCC (std)": grid.cv_results_["std_test_score"],
})

best_idx = grid.best_index_
best_row = table.loc[best_idx]
best_depth = grid.best_params_["max_depth"]
best_crit = grid.best_params_["criterion"]

print()
print(f"{'criterion':>9} | {'max_depth':>9} | {'MCC (mean)':>10} | {'MCC (std)':>9}")
print("-" * 46)
for idx, row in table.iterrows():
    mark = "  <-- best" if idx == best_idx else ""
    print(f"{row['criterion']:>9} | {row['max_depth']:>9} | {row['MCC (mean)']:>10.4f} | {row['MCC (std)']:>9.4f}{mark}")
print("-" * 46)

print(f"\nОптимальні параметри: max_depth = {best_depth}, criterion = '{best_crit}' "
      f"(MCC = {best_row['MCC (mean)']:.4f} ± {best_row['MCC (std)']:.4f})")
print(f"->  TREE_MAX_DEPTH = {best_depth} та TREE_CRITERION = '{best_crit}' вносимо у 03_train_models.py\n")


# MCC vs max_depth для критерія джині та ентропії
depth_labels = [str(d) for d in param_grid["max_depth"]]
x_pos = range(len(depth_labels))

plt.figure(figsize=(8, 5))
for crit, style in [("gini", "bo--"), ("entropy", "go-")]:
    subset = table[table["criterion"] == crit]
    plt.plot(x_pos, subset["MCC (mean)"], style, label=f"criterion = '{crit}'")

plt.title("Залежність якості моделі (MCC) від max_depth та критерію")
plt.xlabel("Максимальна глибина дерева (max_depth)")
plt.ylabel("Метрика MCC")
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(IMAGES_DIR, "tree_depth_vs_mcc.png"), dpi=300)
plt.close()

# топ 15
tree = grid.best_estimator_
importances = pd.Series(tree.feature_importances_, index=X_train.columns)
top15 = importances.sort_values(ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={"width_ratios": [1, 2]})

axes[0].barh(top15.index[::-1], top15.values[::-1])
axes[0].set_title("Top-15 feature importances")
axes[0].set_xlabel("Importance")

plot_tree(tree, max_depth=3, feature_names=X_train.columns,
          class_names=["No", "Yes"], filled=True, fontsize=7, ax=axes[1])
axes[1].set_title("Decision Tree (top 3 levels)")

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_tree_insights.png"), dpi=300)
plt.close()