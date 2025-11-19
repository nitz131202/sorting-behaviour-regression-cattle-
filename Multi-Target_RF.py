# Multi-Target Random Forest (No Aggregation for Targets) 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv").dropna(subset=["diet"])
print(f"Data loaded successfully: {df.shape}")

# Aggregate only sensor-level features per (cow_id, day, diet) 
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

# Flatten multi-level columns
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

# target
target_cols = ["dmi", "av_sorting", "screen1", "screen2", "screen3", "screen4"]
agg_targets = df.groupby(["cow_id", "day", "diet"])[target_cols].first().reset_index()

final_df = pd.merge(agg_df, agg_targets, on=["cow_id", "day", "diet"], how="inner")
print(f"Merged data shape (features + targets): {final_df.shape}")

# Drop rows with missing targets
final_df = final_df.dropna(subset=target_cols)

# X and y
X = final_df.drop(columns=["cow_id", "day", "diet"] + target_cols)
y = final_df[target_cols]

# pipeline
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("rf", MultiOutputRegressor(rf))
])

# Grid Search Parameters
param_grid = {
    "rf__estimator__n_estimators": [100, 300, 500],
    "rf__estimator__max_depth": [10, 20, None],
    "rf__estimator__min_samples_split": [2, 5],
    "rf__estimator__min_samples_leaf": [1, 2]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Run Grid Search 
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

# Evaluate per target using best model 
best_model = grid.best_estimator_
best_model.fit(X, y)
y_pred = best_model.predict(X)

print("\n=== Per-Target Performance on Entire Dataset ===")
for i, target in enumerate(target_cols):
    r2 = r2_score(y.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y.iloc[:, i], y_pred[:, i]))
    print(f"{target:<12}  R² = {r2:>8.4f}   RMSE = {rmse:>8.4f}")

# results
metrics = pd.DataFrame({
    "Target": target_cols,
    "R2": [r2_score(y.iloc[:, i], y_pred[:, i]) for i in range(len(target_cols))],
    "RMSE": [np.sqrt(mean_squared_error(y.iloc[:, i], y_pred[:, i])) for i in range(len(target_cols))]
})
metrics.to_csv("multi_target_results.csv", index=False)
print("\nSaved per-target performance as multi_target_results.csv")
