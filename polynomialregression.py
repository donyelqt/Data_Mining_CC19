# Step 1: Import Required Libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.preprocessing import PolynomialFeatures  # To generate polynomial features
from sklearn.linear_model import LinearRegression  # Linear model for polynomial regression

# Step 2: Create a Dataset (Days vs. COVID-19 Cases)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  
y = np.array([10, 50, 150, 400, 850, 1600, 2900, 5000, 8000, 12000])  

# Step 3: Transform Features to Polynomial Form (Degree = 3)
poly = PolynomialFeatures(degree=3)  
X_poly = poly.fit_transform(X)  

# Step 4: Train the Polynomial Regression Model
model = LinearRegression()  
model.fit(X_poly, y)  

# Step 5: Predict Cases for Day 11
X_test = np.array([11]).reshape(-1, 1)  
X_test_poly = poly.transform(X_test)  
y_pred = model.predict(X_test_poly)  

print(f"Predicted COVID-19 cases on Day 11: {int(y_pred[0])}")

# Step 6: Plot the Data and Regression Curve
plt.scatter(X, y, color='blue', label="Actual Cases")  
plt.plot(X, model.predict(X_poly), color='red', linewidth=2, label="Polynomial Regression")  
plt.xlabel("Days")  
plt.ylabel("COVID-19 Cases")  
plt.legend()  
plt.show()