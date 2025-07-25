## 📈 Logistic Regression

**Logistic Regression** is a **supervised learning algorithm** used for **classification problems**, not regression, despite its name. It is used when the dependent variable is **categorical** (typically binary: 0 or 1, Yes or No, True or False).

---

### 🧠 Why Use Logistic Regression?

- To **predict probabilities** of class membership.
- Best suited for **binary classification** problems like:
  - Spam vs. Not Spam
  - Disease vs. No Disease
  - Purchase vs. No Purchase

---

### 🧮 Mathematical Formula

Instead of fitting a straight line like in linear regression, logistic regression fits a **sigmoid (S-shaped) curve** that outputs values between 0 and 1 (interpreted as probabilities):

\[
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
\]

- Where:
  - \( h_\theta(x) \) is the predicted probability
  - \( \theta \) is the vector of model parameters
  - \( x \) is the feature vector
  - \( e \) is the base of natural logarithms

---

### ✅ Decision Boundary

- If \( h_\theta(x) \geq 0.5 \): predict class `1`
- If \( h_\theta(x) < 0.5 \): predict class `0`

You can adjust the threshold (e.g., 0.3 or 0.7) depending on the use case.

---

### 📊 Cost Function (Log Loss)

The cost function for logistic regression is called **log loss** (or binary cross-entropy):

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]
\]

- \( m \): number of training examples
- \( y^{(i)} \): actual label
- \( h_\theta(x^{(i)}) \): predicted probability

---

### 📌 When to Use Logistic Regression?

- When the **target variable is binary or categorical**.
- When the relationship between features and target is **non-linear** but **linearly separable in probability space**.
- When you want **interpretable coefficients**.

---

### 📋 Summary

| Characteristic           | Description                             |
|--------------------------|-----------------------------------------|
| Output                   | Probabilities (0 to 1)                  |
| Used for                 | Binary classification                   |
| Model function           | Sigmoid / Logistic function             |
| Cost function            | Log Loss (Binary Cross-Entropy)        |
| Assumptions              | Linearity in log-odds                   |
| Extension to multi-class| Softmax Regression (Multinomial Logistic) |

---
