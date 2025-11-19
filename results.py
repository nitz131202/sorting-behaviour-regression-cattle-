# ===== Enhanced Multi-Target Random Forest with Feature Importance + Cowwise Results =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv").dropna(subset=["diet"])
print(f"Data loaded successfully: {df.shape}")

# Aggregate only motion features 
agg_funcs = {
    "total_acc": ["mean", "std"],
    "total_velocity": ["mean", "std"],
    "x+y_cm_moved": ["mean", "std", "sum"],
    "x_acc": ["mean", "std"],
    "x_cm_moved": ["mean", "std", "sum"],
    "x_velocity": ["mean", "std"],
    "y_acc": ["mean", "std"],
    "y_cm_moved": ["mean", "std", "sum"],
    "y_velocity": ["mean", "std"],
    "timediff_sec": ["sum"]
}

agg_df = df.groupby(["cow_id", "day", "diet"]).agg(agg_funcs).reset_index()
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

# target and features
target_cols = ["dmi", "av_sorting", "screen1", "screen2", "screen3", "screen4"]
agg_targets = df.groupby(["cow_id", "day", "diet"])[target_cols].first().reset_index()
final_df = pd.merge(agg_df, agg_targets, on=["cow_id", "day", "diet"], how="inner")
final_df = final_df.dropna(subset=target_cols)
print(f"Merged data shape (features + targets): {final_df.shape}")


X = final_df.drop(columns=["cow_id", "day", "diet"] + target_cols)
y = final_df[target_cols]

# Pipeline 
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("rf", MultiOutputRegressor(rf))
])

# Grid Search Parameters 
param_grid = {
    "rf__estimator__n_estimators": [300],
    "rf__estimator__max_depth": [20],
    "rf__estimator__min_samples_split": [2],
    "rf__estimator__min_samples_leaf": [2]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search 
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring=make_scorer(r2_score, multioutput="uniform_average"),
    n_jobs=-1,
    verbose=2
)
grid.fit(X, y)

print("\nBest parameters found:")
print(grid.best_params_)
print(f"Best cross-validation R² (average across all targets): {grid.best_score_:.4f}")

# Evaluate per-target metrics 
best_model = grid.best_estimator_
best_model.fit(X, y)
y_pred = best_model.predict(X)

print("\n=== Per-Target Performance on Entire Dataset ===")
metrics = []
for i, target in enumerate(target_cols):
    r2 = r2_score(y.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y.iloc[:, i], y_pred[:, i]))
    metrics.append((target, r2, rmse))
    print(f"{target:<12}  R² = {r2:>8.4f}   RMSE = {rmse:>8.4f}")

metrics_df = pd.DataFrame(metrics, columns=["Target", "R2", "RMSE"])
metrics_df.to_csv("multi_target_results.csv", index=False)
print("\nSaved per-target performance as multi_target_results.csv")

# Feature importance analysis 
rf_models = best_model.named_steps["rf"].estimators_
importances = np.mean([est.feature_importances_ for est in rf_models], axis=0)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

feature_importance.to_csv("feature_importance_multi_target_rf.csv", index=False)
print("Saved feature importance as feature_importance_multi_target_rf.csv")

# Plot top 15 features
plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importance.head(15), x="Importance", y="Feature")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = pd.concat([X, y], axis=1).corr()[target_cols].iloc[:-len(target_cols)]
plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature–Target Correlation Heatmap")
plt.tight_layout()
plt.show()

# Cow-wise performance
cowwise = []
for cow_id, group in final_df.groupby("cow_id"):
    X_cow = group.drop(columns=["cow_id", "day", "diet"] + target_cols)
    y_cow = group[target_cols]
    y_pred_cow = best_model.predict(X_cow)

    for i, target in enumerate(target_cols):
        r2 = r2_score(y_cow.iloc[:, i], y_pred_cow[:, i])
        rmse = np.sqrt(mean_squared_error(y_cow.iloc[:, i], y_pred_cow[:, i]))
        cowwise.append([cow_id, target, len(y_cow), r2, rmse])

cowwise_df = pd.DataFrame(cowwise, columns=["cow_id", "Target", "n_samples", "R2", "RMSE"])
cowwise_df.to_csv("cowwise_multi_target_rf.csv", index=False)
print("Saved cowwise performance as cowwise_multi_target_rf.csv")

print("\Done! Generated:")
print("  - multi_target_results.csv  (overall per-target metrics)")
print("  - feature_importance_multi_target_rf.csv  (top predictors)")
print("  - cowwise_multi_target_rf.csv  (per-cow breakdown)")

# Results
import xlsxwriter

output_excel = "multi_target_summary.xlsx"
with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    # Write all sheets
    metrics_df.to_excel(writer, sheet_name="Overall_Results", index=False)
    feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
    cowwise_df.to_excel(writer, sheet_name="Cowwise_Performance", index=False)

    workbook = writer.book

    ws_overall = writer.sheets["Overall_Results"]
    ws_overall.conditional_format("B2:B7", {"type": "3_color_scale"})
    ws_overall.conditional_format("C2:C7", {"type": "3_color_scale"})


    ws_feat = writer.sheets["Feature_Importance"]
    ws_feat.conditional_format("B2:B20", {"type": "3_color_scale"})


    ws_cow = writer.sheets["Cowwise_Performance"]
    # R² column
    ws_cow.conditional_format("D2:D1000", {"type": "3_color_scale"})
    # Highlight RMSE column (reversed: lower is better)
    ws_cow.conditional_format("E2:E1000", {
        "type": "3_color_scale",
        "min_color": "#63BE7B",  # green (good)
        "mid_color": "#FFEB84",
        "max_color": "#F8696B"   # red (bad)
    })

    # Adjust column widths
    for ws in [ws_overall, ws_feat, ws_cow]:
        ws.set_column("A:E", 18)

print(f"\n✨ Combined Excel summary saved as: {output_excel}")
print("   - Sheet 1: Overall_Results")
print("   - Sheet 2: Feature_Importance")
print("   - Sheet 3: Cowwise_Performance")
