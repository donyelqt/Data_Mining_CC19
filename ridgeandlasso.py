import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'SquareFootage': [800, 1200, 1500, 1800, 2200, 2500, 2800, 3200, 3500, 4000, 1000, 2000, 3000, 4000, 5000],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 3, 3, 4, 4, 5],  
    'Age': [10, 5, 15, 20, 8, 12, 7, 3, 25, 30, 5, 5, 5, 5, 5],  # Age of the house
    'DistanceToCity': [10, 8, 12, 5, 7, 4, 6, 3, 2, 1, 1, 2, 3, 4, 5],  # Miles from city center
    'Price': [150000, 180000, 220000, 250000, 280000, 310000, 350000, 390000, 420000, 460000]
}

df = pd.DataFrame(data)

# Define independent (X) and dependent (y) variables
X = df[['SquareFootage', 'Bedrooms', 'Age', 'DistanceToCity']]  # Features
y = df['Price']  # Target variable

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features (Scaling helps in Ridge/Lasso regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=1.0)  # Alpha controls regularization strength
ridge.fit(X_train_scaled, y_train)

# Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Make predictions
ridge_preds = ridge.predict(X_test_scaled)
lasso_preds = lasso.predict(X_test_scaled)

# Compute MSE and R² Score
ridge_mse = mean_squared_error(y_test, ridge_preds)
ridge_r2 = r2_score(y_test, ridge_preds)

lasso_mse = mean_squared_error(y_test, lasso_preds)
lasso_r2 = r2_score(y_test, lasso_preds)

print(f"Ridge Regression - MSE: {ridge_mse:.2f}, R²: {ridge_r2:.2f}")
print(f"Lasso Regression - MSE: {lasso_mse:.2f}, R²: {lasso_r2:.2f}")

# Print coefficients
ridge_coefs = pd.Series(ridge.coef_, index=X.columns)
lasso_coefs = pd.Series(lasso.coef_, index=X.columns)

print("\nRidge Regression Coefficients:")
print(ridge_coefs)

print("\nLasso Regression Coefficients:")
print(lasso_coefs)

plt.figure(figsize=(8, 5))

# Plot Ridge and Lasso coefficients
plt.plot(ridge_coefs, label="Ridge", marker='o')
plt.plot(lasso_coefs, label="Lasso", marker='s')
plt.axhline(0, color='black', linestyle="--", linewidth=1)
plt.legend()
plt.title("Feature Importance: Ridge vs. Lasso")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.show()

 