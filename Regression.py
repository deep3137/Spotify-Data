import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the cleaned data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Define features and target
X = df[['Days', 'Peak Streams', 'Peak Position']]
y = df['Total Streams']

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 5. Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 6. Print performance metrics
print(f"LinearRegression R²: {r2_score(y_test, y_pred_lr):.3f}")
print(f"LinearRegression RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False):,.0f}")
print(f"RandomForestRegressor R²: {r2_score(y_test, y_pred_rf):.3f}")
print(f"RandomForestRegressor RMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):,.0f}")

# 7. Save the actual vs. predicted values
results = pd.DataFrame({
    'actual': y_test.values,
    'pred_lr':  y_pred_lr,
    'pred_rf':  y_pred_rf
})
results.to_csv('regression_results.csv', index=False)
print("\nSaved predictions to regression_results.csv")
