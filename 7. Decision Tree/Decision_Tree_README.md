# 🌳 Decision Tree: Classifier and Regressor

Decision Trees are supervised machine learning models used for both classification and regression tasks. They represent decisions and their consequences using a flowchart-like structure.

---

## 🌲 Decision Tree Classifier

### ✅ Description
A **Decision Tree Classifier** is used when the target variable is **categorical**. The model splits data into branches based on feature values and makes predictions using splitting criteria such as **Gini Impurity** or **Entropy (Information Gain)**.

### 📌 Key Concepts
- **Root Node**: The top node; represents the best feature to split on.
- **Internal Nodes**: Decision nodes that split data.
- **Leaf Nodes**: Final predictions (class labels).
- **Splitting Criteria**:
  - **Gini Index**
  - **Entropy (Information Gain)**

### 📘 Python Implementation

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 📉 Decision Tree Regressor

### ✅ Description
A **Decision Tree Regressor** is used when the target variable is **continuous**. The model splits the data into regions and predicts values by averaging the observations within each leaf.

### 📌 Key Concepts
- **Splitting Criteria**:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
- **Prediction**: Mean value of observations in each region.

### 📘 Python Implementation

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regressor
reg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness of the model.
- **Confusion Matrix**: Actual vs Predicted class comparison.
- **Precision, Recall, F1-Score**: For class-wise performance analysis.

### Regression Metrics
- **Mean Squared Error (MSE)**: Average of squared errors.
- **Mean Absolute Error (MAE)**: Average of absolute errors.
- **R² Score**: Variance explained by the model.

---

## 📈 Optional Visualization

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree Classifier")
plt.show()
```

---

## 🔗 Summary Table

| Task           | Model                  | Splitting Criterion        | Metrics                        |
|----------------|------------------------|-----------------------------|--------------------------------|
| Classification | DecisionTreeClassifier | Gini / Entropy             | Accuracy, F1, Confusion Matrix |
| Regression     | DecisionTreeRegressor  | MSE / MAE                  | MSE, MAE, R² Score             |

---