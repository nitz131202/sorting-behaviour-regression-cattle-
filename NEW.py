
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv").dropna(subset=['diet'])
print(f"Data loaded successfully: {df.shape}")

# Feature Aggregation per (cow_id, day, diet) 
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
    'dmi': ['mean'],  # target variable
}

agg_df = df.groupby(['cow_id', 'day', 'diet']).agg(agg_funcs).reset_index()
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

print(f"Aggregated shape (per cow/day/diet): {agg_df.shape}")

# Drop rows where DMI is missing
agg_df = agg_df.dropna(subset=['dmi_mean'])
print(f"After dropping NaN DMI rows: {agg_df.shape}")
print(agg_df['dmi_mean'].isna().sum(), "rows had missing DMI")

#  Target and Features 
target_variable = 'dmi_mean'
feature_cols = [c for c in agg_df.columns if c not in ['cow_id', 'day', 'diet', target_variable]]

X = agg_df[feature_cols]
y = agg_df[target_variable]
groups = agg_df['cow_id']

# (Hold-Out-One-Cow Validation)
test_cow = 143
train_df = agg_df[agg_df['cow_id'] != test_cow]
test_df  = agg_df[agg_df['cow_id'] == test_cow]

X_train, y_train = train_df[feature_cols], train_df[target_variable]
X_test, y_test   = test_df[feature_cols], test_df[target_variable]

# Pipeline
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

# Train Model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate 
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"\n=== Evaluation on Cow {test_cow} ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Feature Importance 
rf_model = pipeline.named_steps['rf']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Top 20 Important Features ===")
print(importances.head(20))


plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual DMI')
plt.ylabel('Predicted DMI')
plt.title(f'Predicted vs Actual DMI (Cow {test_cow})')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
importances.head(15).plot(kind='barh', color='darkorange')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Summary Table
output_cols = [
    'cow_id', 'day', 'diet',
    'total_acc_mean', 'total_velocity_mean', 'x+y_cm_moved_mean',
    'x_acc_mean', 'x_cm_moved_mean', 'x_velocity_mean',
    'y_acc_mean', 'y_cm_moved_mean', 'y_velocity_mean',
    'timediff_sec_sum', 'x+y_cm_moved_sum', 'x_cm_moved_sum', 'y_cm_moved_sum',
    'total_acc_std', 'total_velocity_std', 'x+y_cm_moved_std',
    'x_acc_std', 'x_cm_moved_std', 'x_velocity_std',
    'y_acc_std', 'y_cm_moved_std', 'y_velocity_std',
    'dmi_mean'
]

agg_df[output_cols].to_csv(f"cow_day_diet_summary.csv", index=False)
print("Summary table saved as cow_day_diet_summary.csv")
