# 🚀 Boosting Techniques in Machine Learning

Boosting is an ensemble technique that builds a strong learner by combining multiple weak learners (typically decision trees) sequentially. Each new model attempts to correct the errors made by previous models.

---

## 🌲 Base Learner: Decision Tree

Boosting techniques commonly use **Decision Trees** as base learners. These trees are usually shallow (stumps with depth=1) and make simple decisions.

### Splitting Criteria

- **Gini Impurity** (used in classification):
  
Gini = 1 - Σ(pᵢ²)

- **Entropy** (Information Gain):
  
Entropy = -Σ(pᵢ * log₂(pᵢ))

Where pᵢ is the probability of class i.

---

## 🔋 AdaBoost (Adaptive Boosting)

### ✅ Description

AdaBoost adjusts the weights of incorrectly predicted samples so that subsequent classifiers focus more on difficult cases.

---

### 🎯 AdaBoost Classification

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost Classifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

### 📈 AdaBoost Regression

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost Regressor
reg = AdaBoostRegressor(n_estimators=50, random_state=42)
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

## 🌄 Gradient Boosting

### ✅ Description

Gradient Boosting improves the model by minimizing the residual errors of previous models using gradient descent. Trees are added sequentially to correct prediction mistakes.

- **Residuals**: Difference between predicted and actual values.
- **Loss Function**: Mean squared error for regression, log loss for classification.

---

### 🎯 Gradient Boosting Classification

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

### 📈 Gradient Boosting Regression

```python
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

## ⚡ XGBoost (Extreme Gradient Boosting)

### ✅ Description

XGBoost is an optimized implementation of Gradient Boosting with additional regularization and speed improvements.

Key terms:

- **Gain**: Improvement in accuracy from a feature split.
- **Cover**: Number of samples affected.
- **Similarity Score**: Measures how well a split separates data.

Install: `pip install xgboost`

---

### 🎯 XGBoost Classification

```python
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

### 📈 XGBoost Regression

```python
from xgboost import XGBRegressor

reg = XGBRegressor(n_estimators=100, learning_rate=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

## 📊 Performance Metrics

### Classification
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

### Regression
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

---

## 🔗 Summary Table

| Task           | Model                      | Evaluation Metrics              | Notes                          |
|----------------|----------------------------|----------------------------------|--------------------------------|
| Classification | AdaBoostClassifier         | Accuracy, F1, Confusion Matrix   | Focuses on difficult samples   |
| Regression     | AdaBoostRegressor          | MSE, MAE, R² Score               | Sequential residual correction |
| Classification | GradientBoostingClassifier | Accuracy, F1, Confusion Matrix   | Learns from residuals          |
| Regression     | GradientBoostingRegressor  | MSE, MAE, R² Score               | Gradient descent on loss       |
| Classification | XGBClassifier              | Accuracy, F1, Confusion Matrix   | Fast, regularized boosting     |
| Regression     | XGBRegressor               | MSE, MAE, R² Score               | Optimized implementation       |

---