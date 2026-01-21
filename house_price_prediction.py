from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['Price'] = housing.target

# Split data
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model with SAFE parameters
model = SGDRegressor(
    max_iter=1,
    warm_start=True,
    eta0=0.001,          # smaller learning rate = stable
    learning_rate="invscaling",
    power_t=0.25,
    random_state=42
)

print("\nTraining started...\n")

# Step-by-step training
for epoch in range(1, 51):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Epoch {epoch} → MSE: {mse:.4f}")

print("\nTraining completed!")

# Convert to dollars
actual_price = y_test.iloc[0] * 100000
predicted_price = predictions[0] * 100000

print("\nExample result (in Dollars):")
print("Actual price: $", round(actual_price, 2))
print("Predicted price: $", round(predicted_price, 2))

