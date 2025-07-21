## 📊 Linear Regression

Linear Regression is one of the most fundamental algorithms in supervised learning. It models the relationship between a **dependent variable (target)** and one or more **independent variables (features)** using a linear function.

---

### 📈 1. Simple Linear Regression

- **Definition**: A linear approach to model the relationship between a **single independent variable (x)** and a **dependent variable (y)**.
- **Equation**:  
  ```
  y = mx + c
  ```  
  or  
  ```
  y = w₀ + w₁x
  ```
- **Goal**: Find the best-fitting straight line that minimizes the difference between predicted and actual values.

---

### 🧮 2. Multiple Linear Regression

- **Definition**: Extension of simple linear regression to include **multiple independent variables**.
- **Equation**:  
  ```
  y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
  ```
- **Goal**: Capture the linear relationship between multiple features and the target.

---

### 📏 3. Performance Metrics for Regression

Used to evaluate how well the regression model fits the data.

| Metric | Description |
|--------|-------------|
| **R² Score (Coefficient of Determination)** | Measures the proportion of variance in the target that is predictable from the features. |
| **MSE (Mean Squared Error)** | Average of the squared differences between actual and predicted values. |
| **RMSE (Root Mean Squared Error)** | Square root of MSE; interpretable in the same units as the target. |
| **MAE (Mean Absolute Error)** | Average of the absolute differences between actual and predicted values. |

---

### 🧪 4. Error Metrics Explained

- **Mean Squared Error (MSE)**:
  ```
  MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
  ```

- **Root Mean Squared Error (RMSE)**:
  ```
  RMSE = √MSE
  ```

- **Mean Absolute Error (MAE)**:
  ```
  MAE = (1/n) * Σ|yᵢ - ŷᵢ|
  ```

These metrics help in understanding how far the predictions deviate from actual values.

---

### ⚠️ 5. Overfitting vs Underfitting

| Concept | Description |
|--------|-------------|
| **Overfitting** | The model learns the noise and details of the training data too well, resulting in poor performance on new data. |
| **Underfitting** | The model is too simple to capture the underlying pattern, leading to poor performance on both training and test data. |

**Solution**: Use cross-validation, regularization (like Ridge or Lasso), and proper model complexity.

---

### 📉 6. Linear Regression using Ordinary Least Squares (OLS)

- **OLS** is the most common method used to estimate the parameters in Linear Regression.
- **Objective**: Minimize the sum of squared residuals (differences between actual and predicted values).
- **Formula** (matrix form):
  ```
  β = (XᵀX)⁻¹ Xᵀy
  ```
  where:
  - `X`: feature matrix
  - `y`: target vector
  - `β`: coefficients

---

### 🔺 7. Polynomial Regression

- **Definition**: A form of regression where the relationship between the independent variable `x` and the dependent variable `y` is modeled as an nth-degree polynomial.
- **Equation**:
  ```
  y = w₀ + w₁x + w₂x² + w₃x³ + ... + wₙxⁿ
  ```
- **Usage**: Useful when data is **non-linearly distributed** but still can be modeled using a continuous curve.
- **Note**: High-degree polynomials can lead to overfitting.

---
