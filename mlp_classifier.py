
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train and evaluate an MLP classifier
def evaluate_model(model, name):
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions) * 100
    test_accuracy = accuracy_score(y_test, test_predictions) * 100

    print(f"\n{name}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    return train_accuracy, test_accuracy


# Part 1: Baseline MLP with one hidden layer and 3 neurons
baseline = MLPClassifier(
    hidden_layer_sizes=(3,),
    random_state=42,
    max_iter=500
)

baseline_train, baseline_test = evaluate_model(
    baseline, "Baseline MLP (3 Neurons)"
)


# Part 2: Hyperparameter tuning (same architecture)
tuned = MLPClassifier(
    hidden_layer_sizes=(3,),
    activation="tanh",
    solver="lbfgs",
    alpha=0.001,
    max_iter=3000,
    random_state=42
)

tuned_train, tuned_test = evaluate_model(
    tuned, "Tuned MLP (3 Neurons)"
)


# Part 3: Increase network complexity
complex_model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="lbfgs",
    alpha=0.001,
    max_iter=3000,
    random_state=42
)

complex_train, complex_test = evaluate_model(
    complex_model, "Complex MLP (10, 10 Neurons)"
)


# Compare results
print("\nClassification Performance Comparison")
print(f"{'Model':<25} {'Training':<15} {'Testing':<15}")
print("-" * 55)
print(f"{'Baseline MLP':<25} {baseline_train:<15.2f} {baseline_test:<15.2f}")
print(f"{'Tuned MLP':<25} {tuned_train:<15.2f} {tuned_test:<15.2f}")
print(f"{'Complex MLP':<25} {complex_train:<15.2f} {complex_test:<15.2f}")

print("\nPerformance Improvement")

print(f"Tuned vs Baseline: {tuned_test - baseline_test:+.2f} percentage points")
print(f"Complex vs Tuned: {complex_test - tuned_test:+.2f} percentage points")
