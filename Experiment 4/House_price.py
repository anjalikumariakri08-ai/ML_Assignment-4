import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 1. Find Housing.csv from the same folder as this Python file
file_path = os.path.join(os.path.dirname(__file__), "Housing.csv")

# 2. Load dataset
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

# 3. Show column names
print("\nColumn names:")
print(df.columns)

# 4. Use ONLY area as input
X = df[["area"]]

# 5. Use price as output
y = df["price"]

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 7. Create Linear Regression model
model = LinearRegression()

# 8. Train the model
model.fit(X_train, y_train)

# 9. Predict house prices
y_pred = model.predict(X_test)

# 10. Display results
print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

# 11. Model evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 12. Take area as user input
area = float(input("\nEnter house area in sq. ft.: "))

# 13. Predict price using ONLY area
new_house = pd.DataFrame({"area": [area]})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])

# 14. Plot the data
plt.scatter(X, y)

# Regression line
plt.plot(X, model.predict(X))

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("House Price Prediction Based on Area")

plt.show()