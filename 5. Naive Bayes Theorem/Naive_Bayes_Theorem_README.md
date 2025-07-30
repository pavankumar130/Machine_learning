# 📘 Naive Bayes: Theorem, Classification, and Regression

## 🔍 Naive Bayes Theorem

Naive Bayes is a supervised learning algorithm based on **Bayes’ Theorem**, with the **“naive” assumption** of conditional independence between features.

### 📐 Bayes' Theorem:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

Where:
- \( P(A|B) \): Posterior probability of class A given predictor B
- \( P(B|A) \): Likelihood of predictor B given class A
- \( P(A) \): Prior probability of class A
- \( P(B) \): Prior probability of predictor B

---

## 🧠 Naive Bayes Classification

Naive Bayes is highly effective for:
- Spam Detection
- Sentiment Analysis
- Document Categorization

### ✅ Popular Variants:
- `GaussianNB`: Works with continuous features
- `MultinomialNB`: Good for count features (e.g., word frequencies)
- `BernoulliNB`: Suitable for binary/boolean features

---

### 📌 Sample Code: Naive Bayes Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
X, y = load_iris(return_X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
## 📉 Naive Bayes Regression (via Discretization)

Naive Bayes is inherently a **classification algorithm**, not designed for predicting continuous outputs. However, **regression tasks can be approximated** using a **discretization-based approach**.

---

### 💡 Concept

1. **Discretize the target variable** (continuous `y`) into bins (e.g., 5–20).
2. Train a `Naive Bayes Classifier` to predict the bin label (classification task).
3. **Inverse-transform** predicted bins to their original numeric range.
4. Evaluate with regression metrics: `MSE`, `MAE`, `R²`.

This method is **not ideal for precise regression**, but it can provide rough estimates when modeling time is crucial or when classification-style processing is needed.

---

### ⚙️ Step-by-Step Python Code

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Step 2: Discretize the continuous target into 10 bins
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Step 4: Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict the bins for test data
y_pred_bins = model.predict(X_test)

# Step 6: Inverse-transform the predicted bins to continuous values
y_pred = discretizer.inverse_transform(y_pred_bins.reshape(-1, 1)).ravel()
y_actual = discretizer.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Step 7: Evaluate using regression metrics
mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")
```
