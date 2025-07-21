# 🧠 Introduction to Machine Learning

---

## 🤖 AI vs ML vs DL vs DS

| Term | Full Form | Description |
|------|-----------|-------------|
| **AI** | Artificial Intelligence | A broad field of computer science focused on building smart machines capable of performing tasks that typically require human intelligence, such as reasoning, learning, and decision-making. |
| **ML** | Machine Learning | A subset of AI that involves algorithms that enable systems to learn patterns from data and make decisions without being explicitly programmed. |
| **DL** | Deep Learning | A specialized branch of ML that uses neural networks with many layers (deep neural networks) to model complex patterns, especially in image, video, and natural language processing tasks. |
| **DS** | Data Science | An interdisciplinary field that uses statistics, ML, data analysis, and domain knowledge to extract meaningful insights from structured and unstructured data. |

### 🔁 Relationship:

Artificial Intelligence
└── Machine Learning
└── Deep Learning
Data Science ↔ Machine Learning

---

## 🧠 Types of Machine Learning Techniques

Machine Learning techniques are broadly categorized into three types:

### 1. ✅ Supervised Learning

- **Definition**: In supervised learning, the model is trained on labeled data (input-output pairs). The goal is to learn a mapping from inputs to outputs.
- **Use Cases**: Spam detection, disease prediction, stock price prediction, etc.

#### 🔍 Common Algorithms:
| Algorithm | Type | Description |
|-----------|------|-------------|
| **Linear Regression** | Regression | Predicts continuous output (e.g., house prices) |
| **Logistic Regression** | Classification | Predicts categorical output (e.g., spam or not) |
| **Decision Trees** | Both | Splits data into branches based on feature values |
| **Random Forest** | Both | Ensemble of decision trees for better accuracy |
| **Support Vector Machines (SVM)** | Both | Finds the optimal hyperplane for classification |
| **K-Nearest Neighbors (KNN)** | Both | Classifies based on closest training samples |
| **Naive Bayes** | Classification | Probabilistic classifier based on Bayes’ theorem |

---

### 2. 🔄 Unsupervised Learning

- **Definition**: The model is trained on **unlabeled data** and tries to identify patterns, groupings, or structure in the data.
- **Use Cases**: Customer segmentation, anomaly detection, recommendation systems.

#### 🔍 Common Algorithms:
| Algorithm | Type | Description |
|-----------|------|-------------|
| **K-Means Clustering** | Clustering | Groups data into k clusters |
| **Hierarchical Clustering** | Clustering | Builds nested clusters in a tree-like structure |
| **DBSCAN** | Clustering | Density-based clustering that handles outliers |
| **Principal Component Analysis (PCA)** | Dimensionality Reduction | Reduces feature space while preserving variance |
| **t-SNE** | Visualization | Reduces dimensions for data visualization |

---

### 3. 🎮 Reinforcement Learning

- **Definition**: An agent learns to take actions in an environment to maximize cumulative reward through trial and error.
- **Use Cases**: Robotics, game playing, self-driving cars, recommendation systems.

#### 🔍 Common Algorithms:
| Algorithm | Type | Description |
|-----------|------|-------------|
| **Q-Learning** | Value-Based | Learns the value of actions in states |
| **SARSA** | Value-Based | Similar to Q-learning but updates values differently |
| **Deep Q Networks (DQN)** | Deep RL | Combines Q-learning with deep neural networks |
| **Policy Gradient Methods** | Policy-Based | Learns a policy directly rather than value functions |
| **Actor-Critic Methods** | Hybrid | Combines value and policy learning |

---
---

## 🧠 Types of Learning Approaches

Machine Learning techniques can also be categorized based on how the model learns from data. The two main approaches are:

### 📌 1. Instance-Based Learning

- **Definition**: The model memorizes the training data and compares new input with stored instances using a similarity measure (like distance).
- **Working**: No explicit generalization is done; predictions are made using raw data at inference time.
- **Pros**: Simple, no training time.
- **Cons**: Slow prediction, memory-intensive, sensitive to noise.

#### 🧪 Examples:
| Algorithm | Description |
|-----------|-------------|
| **K-Nearest Neighbors (KNN)** | Classifies a point based on the majority class among k-nearest data points. |
| **Locally Weighted Regression (LWR)** | Performs regression using a weighted average of nearby data points. |

---

### 🧩 2. Model-Based Learning

- **Definition**: The model learns a function or mapping from input to output during training and uses this learned function to make predictions.
- **Working**: Involves building a model using parameters learned from training data.
- **Pros**: Faster prediction, compact model.
- **Cons**: May require more time for training and can underfit if not tuned well.

#### 🧪 Examples:
| Algorithm | Description |
|-----------|-------------|
| **Linear Regression** | Learns a linear function to predict numeric outcomes. |
| **Logistic Regression** | Predicts probabilities for classification tasks. |
| **Decision Trees** | Splits data recursively to build a predictive model. |
| **Neural Networks** | Uses layers of weights and activations to learn complex patterns. |

---

📘 **Summary:**

| Feature | Instance-Based | Model-Based |
|--------|----------------|-------------|
| Learning Style | Memory-based | Parameter-based |
| Training Time | Fast | Slow (may require optimization) |
| Prediction Time | Slow | Fast |
| Generalization | Lazy (no explicit model) | Generalizes using learned parameters |
| Examples | KNN, LWR | Linear Regression, SVM, Neural Networks |

---