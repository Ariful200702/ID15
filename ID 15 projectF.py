import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'updated_pollution_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Select the features (independent variables) and target (dependent variable)
X = data[['Temperature', 'Humidity']]  # Features
y = data['PM2.5']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Display the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Visualization for linear regression (2D for each feature vs target)
for feature in X.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[feature], y_test, color='blue', alpha=0.5, label='Actual')
    plt.scatter(X_test[feature], y_pred, color='red', alpha=0.5, label='Predicted')
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.title(f'Linear Regression: {feature} vs PM2.5')
    plt.legend()
    plt.show()

