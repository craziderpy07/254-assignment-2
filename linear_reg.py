import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Part 1: Load the provided training dataset
X = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7])
Y = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

# Part 2: Display a scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(X, Y, label="Training data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression - Provided Dataset")

# Part 3: Fit a linear regression model
model = LinearRegression()
model.fit(X.reshape(-1, 1), Y)
Y_pred = model.predict(X.reshape(-1, 1))

# Part 4: Display coefficients and the fitted equation
w0 = model.intercept_
w1 = model.coef_[0]
print("Provided Dataset - Linear Regression")
print(f"Coefficient vector [w0, w1]: [{w0:.6f}, {w1:.6f}]")
print(f"Fitted equation: y = {w0:.6f} + ({w1:.6f})x")

# Part 5: Calculate training RMSE
rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
print(f"Training RMSE: {rmse:.6f}")

order = np.argsort(X)
plt.plot(X[order], Y_pred[order], label="Fitted linear regression")
plt.legend()
plt.tight_layout()

# Part 6: Load GasProperties.csv and use least-squares regression
# Predict Idx using the other four columns as input features.
from pathlib import Path
csv_path = Path(__file__).resolve().parent / "GasProperties.csv"
gas = pd.read_csv(csv_path)
features = ["T", "P", "TC", "SV"]
target = "Idx"
required = features + [target]
if not set(required).issubset(gas.columns):
    raise ValueError(f"GasProperties.csv must contain columns: {required}")
gas = gas[required].apply(pd.to_numeric, errors="coerce").dropna()
X_gas = gas[features].to_numpy(dtype=float)
y_gas = gas[target].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X_gas, y_gas, test_size=0.20, random_state=42
)

# Add a column of ones to include the intercept w0.
X_train_design = np.column_stack((np.ones(len(X_train)), X_train))
X_test_design = np.column_stack((np.ones(len(X_test)), X_test))

# Normal equation: w = (X^T X)^(-1) X^T y
# Solve the equivalent linear system for better numerical stability.
xtx = X_train_design.T @ X_train_design
xty = X_train_design.T @ y_train
try:
    weights = np.linalg.solve(xtx, xty)
except np.linalg.LinAlgError:
    # Pseudoinverse handles singular feature matrices.
    weights = np.linalg.pinv(X_train_design) @ y_train

train_predictions = X_train_design @ weights
test_predictions = X_test_design @ weights
train_rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))

print("\nGasProperties.csv - Least Squares Regression")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("Coefficient vector [w0, w_T, w_P, w_TC, w_SV]:")
print(weights)
print("Fitted equation: Idx = " + " + ".join(
    [f"({weights[0]:.6f})"] +
    [f"({coef:.6f})*{name}" for coef, name in zip(weights[1:], features)]
))
print(f"Training RMSE: {train_rmse:.6f}")
print(f"Testing RMSE: {test_rmse:.6f}")

plt.show()
