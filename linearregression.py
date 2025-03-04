from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([800, 1200, 1500, 1800, 2200]).reshape(-1,1)  # Independent variable (square footage)
y = np.array([150000, 180000, 220000, 250000, 280000])  # Dependent variable (price)

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict([[2000]])  # Predict price for 2000 sqft house
print("Predicted Price:", y_pred[0])

# Create plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')  # Plot actual data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot regression line
plt.scatter([2000], y_pred, color='green', label=f'Prediction (2000 sqft: ${y_pred[0]:.0f})')  # Plot prediction point

# Customize plot
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Price vs Square Footage')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.show()