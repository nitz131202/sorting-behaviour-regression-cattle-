
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv").dropna(subset=['diet'])
print(f"Data loaded successfully: {df.shape}")

# === Feature aggregation per (cow_id, day, diet) ===
agg_funcs = {
    'total_acc': ['mean', 'std'],
    'total_velocity': ['mean', 'std'],
    'x+y_cm_moved': ['mean', 'std', 'sum'],
    'x_acc': ['mean', 'std'],
    'x_cm_moved': ['mean', 'std', 'sum'],
    'x_velocity': ['mean', 'std'],
    'y_acc': ['mean', 'std'],
    'y_cm_moved': ['mean', 'std', 'sum'],
    'y_velocity': ['mean', 'std'],
    'timediff_sec': ['sum'],
    'dmi': ['mean'],  # target
}

agg_df = df.groupby(['cow_id', 'day', 'diet']).agg(agg_funcs).reset_index()
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
print(f"Aggregated shape (per cow/day/diet): {agg_df.shape}")

# === Drop rows with missing DMI ===
agg_df = agg_df.dropna(subset=['dmi_mean'])
print(f"After dropping NaN DMI rows: {agg_df.shape}")
print(agg_df['dmi_mean'].isna().sum(), "rows had missing DMI")

# === Rename "_mean" to "_av" for readability ===
agg_df = agg_df.rename(columns=lambda x: x.replace('_mean', '_av'))

# === Define features and target ===
target_variable = 'dmi_av'
feature_cols = [c for c in agg_df.columns if c not in ['cow_id', 'day', 'diet', target_variable]]

results = []

# === Define pipeline ===
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# === Leave-One-Cow-Out Cross-Validation ===
unique_cows = agg_df['cow_id'].unique()

for test_cow in unique_cows:
    train_df = agg_df[agg_df['cow_id'] != test_cow]
    test_df  = agg_df[agg_df['cow_id'] == test_cow]
    
    X_train, y_train = train_df[feature_cols], train_df[target_variable]
    X_test, y_test   = test_df[feature_cols], test_df[target_variable]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if len(y_test) == 1:
        r2 = float('nan')
        rmse = sqrt(mean_squared_error(y_test, y_pred))
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        'cow_id': test_cow,
        'n_samples': len(test_df),
        'R2': r2,
        'RMSE': rmse,
        'mean_actual_DMI': y_test.mean(),
        'mean_predicted_DMI': y_pred.mean()
    })

    print(f"Cow {test_cow}: R²={r2:.4f} RMSE={rmse:.4f} (n={len(test_df)})")

# === Results summary ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='cow_id')
print("\n=== Leave-One-Cow-Out Results ===")
print(results_df)

print(f"\nAverage R²: {results_df['R2'].mean():.4f}")
print(f"Average RMSE: {results_df['RMSE'].mean():.4f}")

results_df.to_csv("cowwise_results.csv", index=False)
print("\nSaved per-cow performance as cowwise_results.csv")

# === Export the final daily summary table ===
output_cols = [
    'cow_id', 'day', 'diet',
    'total_acc_av', 'total_velocity_av', 'x+y_cm_moved_av',
    'x_acc_av', 'x_cm_moved_av', 'x_velocity_av',
    'y_acc_av', 'y_cm_moved_av', 'y_velocity_av',
    'timediff_sec_sum', 'x+y_cm_moved_sum', 'x_cm_moved_sum', 'y_cm_moved_sum',
    'total_acc_std', 'total_velocity_std', 'x+y_cm_moved_std',
    'x_acc_std', 'x_cm_moved_std', 'x_velocity_std',
    'y_acc_std', 'y_cm_moved_std', 'y_velocity_std',
    'dmi_av'
]

agg_df[output_cols].to_csv("cow_day_diet_summary.csv", index=False)
print("Summary table saved as cow_day_diet_summary.csv")

# === Visualization: R² per cow ===
plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x='cow_id', y='R2', color='steelblue')
plt.axhline(0, color='red', linestyle='--')
plt.title("Leave-One-Cow-Out R² per Cow")
plt.ylabel("R²")
plt.xlabel("Cow ID")
plt.tight_layout()
plt.show()