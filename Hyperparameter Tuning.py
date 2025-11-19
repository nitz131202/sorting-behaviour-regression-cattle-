
# Random Forest Hyperparameter Tuning (using DMI)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


df = pd.read_csv("/Users/sreenithyam/Desktop/New Project/main_dataframe.csv").dropna(subset=['diet'])
print(f"Data loaded successfully: {df.shape}")

# Feature aggregation 
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


agg_df = agg_df.rename(columns={'dmi_mean': 'dmi'})
agg_df = agg_df.dropna(subset=['dmi'])
print(f"After dropping NaN DMI rows: {agg_df.shape}")

# features and target 
X = agg_df.drop(columns=['cow_id', 'day', 'diet', 'dmi'])
y = agg_df['dmi']

# pipeline 
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# hyperparameter grid 
param_grid = {
    'rf__n_estimators': [100, 200, 300, 500],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search 
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid.fit(X, y)


print("\nBest parameters:", grid.best_params_)
print("Best cross-validation R²:", grid.best_score_)
