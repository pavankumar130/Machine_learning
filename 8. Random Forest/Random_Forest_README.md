# 🌲 Random Forest: Classifier and Regressor

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting. It works by constructing a multitude of decision trees during training and outputs the mode (classification) or mean (regression) of the individual trees.

---

## 🌳 Random Forest Classifier

### ✅ Description
A **Random Forest Classifier** is used when the target variable is **categorical**. It builds multiple decision trees using random subsets of features and samples, then aggregates the predictions using majority voting.

### 📌 Key Concepts
- **Ensemble of Trees**: Combines multiple decision trees.
- **Bootstrap Aggregation (Bagging)**: Training on random samples with replacement.
- **Feature Randomness**: Each tree considers a random subset of features.
- **Out-of-Bag (OOB) Evaluation**: Unused samples help validate model accuracy without a separate validation set.

### 📘 Python Implementation

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with OOB evaluation
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("OOB Score:", clf.oob_score_)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 📉 Random Forest Regressor

### ✅ Description
A **Random Forest Regressor** is used when the target variable is **continuous**. It predicts by averaging the results of individual trees, each trained on random samples and feature subsets.

### 📌 Key Concepts
- **Ensemble Averaging**: Predictions are averaged across all trees.
- **Reduces Overfitting**: Compared to individual decision trees.
- **OOB Evaluation**: An internal validation method using unused bootstrap samples.

### 📘 Python Implementation

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor with OOB score
reg = RandomForestRegressor(n_estimators=100, oob_score=True, bootstrap=True, random_state=42)
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# Note: OOB score in regression is like R^2 on OOB samples
print("OOB Score:", reg.oob_score_)
```

---

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness of the model.
- **Confusion Matrix**: Actual vs predicted classes.
- **OOB Score**: Internal cross-validation performance.
- **Precision, Recall, F1-Score**: For class-wise performance.

### Regression Metrics
- **Mean Squared Error (MSE)**: Average squared difference.
- **Mean Absolute Error (MAE)**: Average absolute difference.
- **R² Score**: Variance explained by the model.
- **OOB Score**: Similar to R² using unused bootstrap samples.

---

## 🔗 Summary Table

| Task           | Model                  | Evaluation Criterion        | Extra Evaluation        |
|----------------|------------------------|------------------------------|--------------------------|
| Classification | RandomForestClassifier | Accuracy, F1, Confusion Matrix | OOB Score                |
| Regression     | RandomForestRegressor  | MSE, MAE, R² Score             | OOB Score                |

---