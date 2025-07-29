
# 📘 Support Vector Machines (SVM) in Python

This guide walks you through:

- Support Vector Classifier (SVC)
- Support Vector Regression (SVR)
- Understanding and Visualizing SVM Kernels
- Performance Metrics for Model Evaluation

---

## 📌 1. Support Vector Classifier (SVC)

### 🧠 Description:
Support Vector Classifier is a supervised learning model used for binary classification tasks. It finds the optimal hyperplane that separates the data into two classes with maximum margin.

### 🧪 Dataset:
Synthetic dataset using `make_classification`.

### ✅ Python Code:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_clusters_per_class=2, n_redundant=0)

# Visualize data
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Train and Evaluate for different kernels
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nKernel: {kernel}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
```

### 🔍 Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)

# Final evaluation
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📌 2. Support Vector Regression (SVR)

### 🧠 Description:
Support Vector Regression is used for continuous target prediction while maintaining robustness to outliers.

### 🧪 Dataset:
`tips` dataset from Seaborn.

### ✅ Python Code:
```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
df = sns.load_dataset('tips')
X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = df['total_bill']

# Encode categorical features
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

X['sex'] = le1.fit_transform(X['sex'])
X['smoker'] = le2.fit_transform(X['smoker'])
X['time'] = le3.fit_transform(X['time'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# One-hot encode 'day'
ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Train SVR
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
```

### 🔍 Hyperparameter Tuning:
```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVR(), param_grid, verbose=3, refit=True)
grid.fit(X_train, y_train)

grid_pred = grid.predict(X_test)
print("Best Params:", grid.best_params_)
print("R² Score:", r2_score(y_test, grid_pred))
print("MAE:", mean_absolute_error(y_test, grid_pred))
```

---

## 📌 3. SVM Kernels (Visualization & Intuition)

### 🧠 Description:
SVM Kernels transform non-linearly separable data into higher dimensions where it becomes linearly separable.

### 📊 Kernel Types:
- **Linear**
- **Polynomial**
- **RBF (Radial Basis Function)**
- **Sigmoid**

### ✅ Python Code (Kernel Comparison):
```python
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create concentric circles (non-linear pattern)
x = np.linspace(-5, 5, 100)
y = np.sqrt(10**2 - x**2)
y = np.hstack([y, -y])
x = np.hstack([x, -x])
df1 = pd.DataFrame(np.vstack([y, x]).T, columns=['X1', 'X2'])
df1['Y'] = 0

x1 = np.linspace(-5, 5, 100)
y1 = np.sqrt(5**2 - x1**2)
y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])
df2 = pd.DataFrame(np.vstack([y1, x1]).T, columns=['X1', 'X2'])
df2['Y'] = 1

df = df1.append(df2)

# Polynomial kernel feature expansion
df['X1_Square'] = df['X1']**2
df['X2_Square'] = df['X2']**2
df['X1*X2'] = df['X1'] * df['X2']

X = df[['X1', 'X2', 'X1_Square', 'X2_Square', 'X1*X2']]
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Kernel: {kernel} => Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 📊 Performance Metrics

| Metric                  | Classifier Task            | Regression Task             |
|-------------------------|----------------------------|-----------------------------|
| Accuracy                | ✅                          | ❌                          |
| Precision, Recall, F1   | ✅                          | ❌                          |
| Confusion Matrix        | ✅                          | ❌                          |
| R² Score                | ❌                          | ✅                          |
| Mean Absolute Error     | ❌                          | ✅                          |
| GridSearchCV (Tuning)   | ✅                          | ✅                          |

---

## 📎 Summary

| SVM Type | Use Case        | Key Module | Kernel Support |
|----------|------------------|------------|----------------|
| SVC      | Classification    | `sklearn.svm.SVC` | Linear, RBF, Poly, Sigmoid |
| SVR      | Regression        | `sklearn.svm.SVR` | Linear, RBF, Poly, Sigmoid |
