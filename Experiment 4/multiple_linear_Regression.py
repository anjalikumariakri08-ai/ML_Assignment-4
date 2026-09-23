import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 1. Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "Housing.csv")
df = pd.read_csv(file_path)

# 2. Display first 5 rows
print("First 5 rows:")
print(df.head())

# 3. Select TWO features
#    Area and Number of Bedrooms
X = df[["area", "bedrooms"]]

# 4. Select price as target
y = df["price"]

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Create Multiple Linear Regression model
model = LinearRegression()

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict house prices
y_pred = model.predict(X_test)

# 9. Display coefficients
print("\nCoefficients:")
print("Area coefficient:", model.coef_[0])
print("Bedrooms coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

# 10. Evaluate the model
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 11. Take input from user
area = float(input("\nEnter house area in sq. ft.: "))
bedrooms = int(input("Enter number of bedrooms: "))

# 12. Predict price
new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms]
})

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])