# 📄 README — Principal Component Analysis (PCA)

## 📌 Overview
**Principal Component Analysis (PCA)** is a powerful **unsupervised dimensionality reduction** technique.  
It transforms the original correlated features into a smaller set of **uncorrelated variables** (principal components) while **retaining most of the variation** in the dataset.

**Applications:**
- Data compression
- Noise reduction
- Visualization of high-dimensional data
- Preprocessing for machine learning models

---

## 1️⃣ Intuition

Imagine you have a cloud of points in high-dimensional space:

- PCA **rotates the coordinate system** to align with the directions where the data varies most.
- The **first principal component (PC1)** points along the direction of **maximum variance**.
- The **second principal component (PC2)** is orthogonal to PC1 and captures the next highest variance, and so on.
- By **keeping only the first few PCs**, we reduce dimensionality while keeping the essential structure of the data.

💡 **Analogy**: Think of shining a light on a 3D object to make a 2D shadow — you want to choose the angle so that the shadow keeps as much detail as possible.

---

## 2️⃣ Mathematical Derivation

Let \( X \) be our data matrix of shape \((n_{\text{samples}}, n_{\text{features}})\).

### Step 1: Standardization  
We center (and often scale) the data:

\[
X_{\text{centered}} = X - \mu
\]
where \( \mu \) is the mean of each column (feature).

---

### Step 2: Covariance Matrix  
The covariance matrix tells us how features vary together:

\[
\Sigma = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
\]
Shape: \((n_{\text{features}} \times n_{\text{features}})\)

---

### Step 3: Eigen Decomposition  
We solve:

\[
\Sigma v = \lambda v
\]
- \( v \): eigenvector → principal component direction  
- \( \lambda \): eigenvalue → variance magnitude along that direction  

**Properties**:
- Eigenvectors are orthogonal.
- Eigenvalues are non-negative.
- Largest eigenvalue → most important component.

---

### Step 4: Sorting & Selection  
We sort eigenvalues in descending order and select the top \( k \) eigenvectors to form the projection matrix:

\[
W = [v_1, v_2, \dots, v_k]
\]

---

### Step 5: Projection  
The reduced dataset:

\[
Z = X_{\text{centered}} W
\]
Shape: \((n_{\text{samples}} \times k)\)

---

## 3️⃣ PCA via Singular Value Decomposition (SVD)  

Instead of covariance matrix, PCA can be computed directly using **SVD**:

\[
X_{\text{centered}} = U S V^T
\]
- Columns of \( V \) are the **principal components**.
- \( S^2 / (n-1) \) gives eigenvalues.

**Advantages of SVD**:
- More numerically stable than covariance eigen-decomposition.
- Works well for large datasets.

---

## 4️⃣ Variance Explained

The **explained variance ratio** tells us how much information each PC retains:

\[
\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}
\]

Example:
If PC1 = 73%, PC2 = 23% → together they explain 96% of total variance.

---

## 5️⃣ PCA vs. LDA

| Aspect              | PCA                                        | LDA                                         |
|---------------------|--------------------------------------------|----------------------------------------------|
| Supervision         | Unsupervised                               | Supervised                                   |
| Goal                | Maximize variance                          | Maximize class separability                  |
| Uses Labels?        | ❌ No                                      | ✅ Yes                                       |
| Projection          | Based on eigenvectors of covariance matrix | Based on eigenvectors of scatter matrices    |
| When to use         | Dimensionality reduction, visualization    | Classification, supervised feature extraction|

---

## 6️⃣ Python Implementation

```python
"""
Principal Component Analysis (PCA) - Complete Example
------------------------------------------------------
1. Standardize data
2. Compute covariance matrix
3. Eigen decomposition OR use SVD
4. Sort components by variance
5. Project data to lower dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Step 1: Standardization
X_std = StandardScaler().fit_transform(X)

# Step 2: Covariance matrix
cov_matrix = np.cov(X_std.T)

# Step 3: Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort by variance
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Step 5: Select top k components
k = 2
W = eigenvectors[:, :k]

# Step 6: Project data
X_pca = X_std @ W

# Step 7: Explained variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio[:k])

# Step 8: Plot result
plt.figure(figsize=(8,6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=data.target_names[label])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.title("PCA Projection of Iris Dataset")
plt.show()
```

---

## 7️⃣ Practical Tips

- Always **standardize** before PCA (especially when features have different scales).
- PCA is sensitive to **outliers** (large variance can be caused by them).
- Choose \( k \) using **cumulative explained variance plot**.
- For huge datasets, use **IncrementalPCA** or **RandomizedPCA** in scikit-learn.

---

## 8️⃣ Cumulative Variance Plot (Scree Plot)

```python
# Scree plot
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.show()
```

---

## 📌 Summary

- PCA **rotates** data to find orthogonal directions of maximum variance.
- Eigen decomposition or SVD can be used.
- Keep components until cumulative variance > 90–95%.
- Use PCA for **compression, noise reduction, visualization**, but not for supervised class separation (use LDA instead).
