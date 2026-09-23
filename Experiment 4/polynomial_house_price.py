import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os
file_path = os.path.join(os.path.dirname(__file__), "Housing.csv")
df = pd.read_csv(file_path)
X = df[["area"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, y_pred_linear)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)
y_pred_poly = polynomial_model.predict(X_test_poly)
polynomial_r2 = r2_score(y_test, y_pred_poly)
print("Standard Linear Regression R2 Score:")
print(linear_r2)
print("\nPolynomial Regression R2 Score:")
print(polynomial_r2)
print("\nComparison:")
if polynomial_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 score.")
elif polynomial_r2 < linear_r2:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")
X_plot = pd.DataFrame({
    "area": sorted(df["area"])
})
X_plot_poly = poly.transform(X_plot)
y_plot = polynomial_model.predict(X_plot_poly)
plt.scatter(X, y)
plt.plot(X_plot, y_plot)
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("Polynomial Regression: Area vs House Price")
plt.show()
