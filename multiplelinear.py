import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset
data = {
    'SquareFootage': [800, 1200, 1500, 1800, 2200],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [150000, 180000, 220000, 250000, 280000]
}
df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['SquareFootage', 'Bedrooms']]
y = df['Price']

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot training data points
ax.scatter(X_train['SquareFootage'], X_train['Bedrooms'], y_train, 
          c='blue', label='Training Data')
# Plot testing data points
ax.scatter(X_test['SquareFootage'], X_test['Bedrooms'], y_test, 
          c='green', label='Test Data')
# Plot predicted points
ax.scatter(X_test['SquareFootage'], X_test['Bedrooms'], y_pred, 
          c='red', label='Predictions')

# Create meshgrid for regression plane
x_surf = np.linspace(min(df['SquareFootage']), max(df['SquareFootage']), 20)
y_surf = np.linspace(min(df['Bedrooms']), max(df['Bedrooms']), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf

# Plot regression plane
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='gray')

# Customize plot
ax.set_xlabel('Square Footage')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price ($)')
ax.set_title('House Price Prediction: Square Footage vs Bedrooms vs Price')
ax.legend()

# Adjust view angle for better visualization
ax.view_init(elev=20, azim=-45)

plt.show()