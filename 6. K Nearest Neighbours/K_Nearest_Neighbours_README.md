# 🔍 K-Nearest Neighbours (KNN)

K-Nearest Neighbours (KNN) is a **non-parametric**, **instance-based** learning algorithm used for both **classification** and **regression**.

It works by:
- Storing all training data
- Making predictions based on the majority (for classification) or average (for regression) of the `k` nearest data points

---

## 🧠 KNN Classification

KNN classification assigns a class label to a data point based on the **majority class among its `k` nearest neighbors**.

### ✅ Sample Code: KNN Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
X, y = load_iris(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## 📉 K-Nearest Neighbours Regression

KNN Regression is a **non-parametric, lazy learning** algorithm that predicts the target value of a data point by taking the **average (or weighted average)** of the `k` nearest training data points.

### 🧠 How It Works

1. Choose the number of neighbors `k`.
2. Compute the distance (typically Euclidean) between the test point and all training points.
3. Select the `k` closest points.
4. Predict the value as the **mean** (or weighted mean) of these `k` points' target values.

---

### ✅ Sample Code: KNN Regression with Evaluation

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train KNN Regressor
model = KNeighborsRegressor(n_neighbors=5)  # Try different values of k
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")
```
