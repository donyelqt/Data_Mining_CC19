from sklearn.linear_model import LinearRegression
import numpy as np

# Sample dataset
X = np.array([800, 1200, 1500, 1800, 2200]).reshape(-1,1)  # Independent variable (square footage)
y = np.array([150000, 180000, 220000, 250000, 280000])  # Dependent variable (price)

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict([[2000]])  # Predict price for 2000 sqft house
print("Predicted Price:", y_pred[0])