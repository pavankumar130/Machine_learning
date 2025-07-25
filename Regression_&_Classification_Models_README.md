# 📘 Regression & Classification Models - README

This repository contains explanations and Python implementations of several regression and classification models used in machine learning.

## 📚 Table of Contents

- [📊 1. Linear Regression](#-1-linear-regression)
- [📊 2. Multiple Linear Regression](#-2-multiple-linear-regression)
- [📈 3. Polynomial Regression](#-3-polynomial-regression)
- [📉 4. Ridge Regression](#-4-ridge-regression)
- [📉 5. Lasso Regression](#-5-lasso-regression)
- [📉 6. ElasticNet Regression](#-6-elasticnet-regression)
- [📈 Logistic Regression](#-logistic-regression)

---


## 📊 1. Linear Regression

**What:** Models the relationship between one independent and one dependent variable using a straight line.

**Why:** Use when there's a linear relationship and only one feature.

### 📌 Python Code
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = df[['feature']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

```
## 📊 2. Multiple Linear Regression

### 📌 What is it?
Multiple Linear Regression is an extension of simple linear regression that models the relationship between **two or more independent variables** and one **dependent variable** by fitting a linear equation.

### ✅ Why do we use it?
We use multiple linear regression when:
- There are **multiple features** affecting the target.
- The relationship between predictors and the outcome is **linear**.
- We want to **quantify** the influence of each feature on the outcome.

### 📏 Performance Metrics
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors.
- **MSE (Mean Squared Error)**: Squares the error to penalize larger differences.
- **RMSE (Root Mean Squared Error)**: The square root of MSE; interpretable in the same units as the target.
- **R² Score**: Measures how well the model explains the variability of the target.

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sample DataFrame
# df = pd.read_csv('your_dataset.csv')

# Assume df has multiple features and a target column
X = df[['feature1', 'feature2', 'feature3']]  # Replace with actual feature names
y = df['target']  # Replace with your actual target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
```

## 📈 3. Polynomial Regression

### 📌 What is it?
Polynomial Regression is a type of regression that models the relationship between the independent variable `x` and the dependent variable `y` as an `n`-th degree polynomial. It is useful when the data shows **non-linear** trends.

Unlike Linear Regression, it allows for curvature in the model.

### ✅ Why do we use it?
- When data shows **non-linear trends** that linear models can't capture.
- When residual plots of linear regression indicate curvature.
- To improve performance on datasets where features have exponential, quadratic, or cubic effects on the target.

---

### 📏 Performance Metrics
- **Mean Absolute Error (MAE)**: Average of the absolute errors.
- **Mean Squared Error (MSE)**: Average of squared errors; penalizes large errors.
- **Root Mean Squared Error (RMSE)**: Square root of MSE.
- **R² Score**: Proportion of the variance in the dependent variable that's predictable.

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
# df = pd.read_csv('your_dataset.csv')
X = df[['feature']]  # Replace with your actual feature
y = df['target']     # Replace with your target column

# Create polynomial features
poly = PolynomialFeatures(degree=3)  # You can change the degree
X_poly = poly.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```

## 📉 4. Ridge Regression

### 📌 What is it?
Ridge Regression is a type of regularized linear regression that adds a **penalty (L2 regularization)** to the loss function to reduce model complexity and prevent overfitting.

Where:
- RSS = Residual Sum of Squares
- \(\alpha\) = regularization strength
- \(\beta_j\) = model coefficients

---

### ✅ Why do we use it?
- When the model **overfits** due to **multicollinearity** or too many features.
- Helps in **shrinking** coefficients to prevent them from becoming too large.
- Ideal when **all features are relevant** but need small adjustments.

---

### 📏 Performance Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sample dataset
# df = pd.read_csv('your_dataset.csv')
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your actual features
y = df['target']  # Replace with your actual target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge model
ridge = Ridge(alpha=1.0)  # You can tune alpha
ridge.fit(X_train, y_train)

# Predictions
y_pred = ridge.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```

## 📉 5. Lasso Regression

### 📌 What is it?
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a regularized regression technique that adds an **L1 penalty** to the loss function. It not only prevents overfitting but also performs **feature selection** by reducing some coefficients to exactly zero.

Where:
- RSS = Residual Sum of Squares
- \(\alpha\) = regularization strength
- \(\beta_j\) = model coefficients

---

### ✅ Why do we use it?
- Useful when we want **sparse models** (i.e., some coefficients = 0).
- Helps in **feature selection**, especially when many features are irrelevant.
- Controls **overfitting** in high-dimensional datasets.

---

### 📏 Performance Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sample dataset
# df = pd.read_csv('your_dataset.csv')
X = df[['feature1', 'feature2', 'feature3']]  # Replace with actual features
y = df['target']  # Replace with target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Lasso model
lasso = Lasso(alpha=0.1)  # You can tune alpha
lasso.fit(X_train, y_train)

# Predictions
y_pred = lasso.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```

## 📉 6. ElasticNet Regression

### 📌 What is it?
ElasticNet Regression combines **L1 (Lasso)** and **L2 (Ridge)** regularization. It aims to get the **feature selection** benefits of Lasso and the **shrinkage** benefits of Ridge.

Where:
- RSS = Residual Sum of Squares
- \(\alpha\) = overall regularization strength
- \(\lambda\) = mixing parameter between Lasso (L1) and Ridge (L2)

---

### ✅ Why do we use it?
- Useful when:
  - There are **many features** (high dimensionality).
  - There’s **correlation among features**.
  - We want both **feature selection** and **coefficient shrinkage**.
- Overcomes limitations of pure Lasso and pure Ridge.

---

### 📏 Performance Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sample dataset
# df = pd.read_csv('your_dataset.csv')
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your features
y = df['target']  # Replace with your target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ElasticNet model
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio = balance between Lasso and Ridge
elastic.fit(X_train, y_train)

# Predictions
y_pred = elastic.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```

## 📈 Logistic Regression

### 📌 What is it?
Logistic Regression is a **classification** algorithm used to predict **binary** or **multi-class** outcomes. It models the probability that a given input point belongs to a particular category using the **logistic (sigmoid)** function.


It outputs values between **0 and 1**, which can be interpreted as **probabilities**.

---

### ✅ Why do we use it?
- Used for **classification tasks**, such as:
  - Spam vs. not spam
  - Disease vs. no disease
  - Click vs. no click
- Simple and fast model.
- Works well when features and the outcome have a **linear decision boundary**.

---

### 📏 Performance Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC AUC Score**

---

### 🐍 Python Code Example
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Sample dataset
# df = pd.read_csv('your_dataset.csv')
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your features
y = df['target']  # Target should be binary (0 or 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {auc}")
print(f"Confusion Matrix:\n{cm}")
```
