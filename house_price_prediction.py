import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("house_data.csv")

# Features (X) and Target (y)
X = data[["Rooms", "Size"]]
y = data["Price"]

# Split data (training & testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Print results
print("Predicted Prices:", predictions)
print("Actual Prices:", y_test.values)

# Example prediction
new_house = [[3, 1300]]  # 3 rooms, 1300 size
predicted_price = model.predict(new_house)
print("\nPrice for 3 rooms & 1300 size:", predicted_price[0])

# Plot (Size vs Price)
plt.scatter(data["Size"], data["Price"])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()