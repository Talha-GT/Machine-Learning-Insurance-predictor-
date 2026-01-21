

import pandas as pd
import numpy as np

# ML tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("insurance.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset Shape:", df.shape)


df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['region'], drop_first=True)

df = df[df['bmi'] < 50]

X = df.drop('charges', axis=1)
y = df['charges']

print("\nFeatures:")
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])


cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='r2'
)

print("\nCross Validation Results:")
print("Mean R2:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [2, 3, 4]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest Set Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

import pickle

with open("gb_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved as gb_model.pkl")
