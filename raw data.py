
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv")
df = df.dropna(subset=['diet', 'dmi'])
print(f"Data loaded successfully: {df.shape}")


selected_features = [
    'mouth_x', 'mouth_y',
    'x_cm', 'y_cm',
    'timediff_sec',
    'x_cm_moved', 'y_cm_moved', 'x+y_cm_moved',
    'x_velocity', 'y_velocity', 'total_velocity',
    'x_acc', 'y_acc', 'total_acc',
    'day'
]

# Keep only columns that actually exist in df
selected_features = [f for f in selected_features if f in df.columns]

target = 'dmi'

X = df[selected_features]
y = df[target]
groups = df['cow_id']

print(f"Using {len(selected_features)} features:")
print(selected_features)


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    ))
])


# Leave-One-Cow-Out cross-validation

unique_cows = df['cow_id'].unique()
results = []

for test_cow in unique_cows:
    train_df = df[df['cow_id'] != test_cow]
    test_df  = df[df['cow_id'] == test_cow]

    X_train, y_train = train_df[selected_features], train_df[target]
    X_test,  y_test  = test_df[selected_features],  test_df[target]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # metrics
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        'cow_id': test_cow,
        'n_samples': len(y_test),
        'R2': r2,
        'RMSE': rmse,
        'mean_actual_DMI': y_test.mean(),
        'mean_predicted_DMI': y_pred.mean()
    })

    print(f"Cow {test_cow}: R²={r2:.4f} RMSE={rmse:.4f} (n={len(y_test)})")


# results

results_df = pd.DataFrame(results)
print("\n=== Leave-One-Cow-Out Results ===")
print(results_df)

print(f"\nAverage R²: {np.nanmean(results_df['R2']):.4f}")
print(f"Average RMSE: {results_df['RMSE'].mean():.4f}")

results_df.to_csv("cowwise_results_motion_only.csv", index=False)
print("Saved per-cow performance as cowwise_results_motion_only.csv")



rf_model = pipeline.named_steps['rf']
importances = pd.Series(rf_model.feature_importances_, index=selected_features).sort_values(ascending=False)

plt.figure(figsize=(8,6))
importances.head(10).plot(kind='barh', color='royalblue')
plt.title('Top 10 Feature Importances (Motion/Distance Only)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 features:")
print(importances.head(10))
