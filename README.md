# 🧠 MLP from Scratch Using NumPy

A simple implementation of a **Multi-Layer Perceptron (MLP)** built from scratch using **NumPy** for binary classification on the **Two Moons** dataset. This project demonstrates the core concepts of neural networks, including forward propagation, backpropagation, gradient descent, and decision boundary visualization without using deep learning frameworks like TensorFlow or PyTorch.

---

## 📌 Objective

The objective of this project is to understand how a Multi-Layer Perceptron (MLP) works internally by implementing every component manually using NumPy. This project serves as a foundation for learning deep learning concepts before moving to frameworks such as TensorFlow or PyTorch.

---

## ✨ Features

* Multi-Layer Perceptron implemented from scratch
* Forward Propagation
* Backpropagation using Gradient Descent
* Sigmoid Activation Function
* Binary Cross-Entropy Loss
* Two Moons Dataset Classification
* Feature Scaling with StandardScaler
* Decision Boundary Visualization
* Training Loss Curve
* Test Accuracy Evaluation

---

## 🛠️ Tech Stack

* Python 3.11+
* NumPy
* Matplotlib
* Scikit-learn
* UV (Python Package Manager)
* Visual Studio Code

---

## 📂 Project Structure

```text
MLP-NumPy/
│
├── .venv/
├── mlp_numpy.py
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MLP-NumPy.git
cd MLP-NumPy
```

### 2. Create Virtual Environment

```bash
uv venv
```

### 3. Activate Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
uv add numpy matplotlib scikit-learn
```

---

## ▶️ Run the Project

```bash
uv run mlp_numpy.py
```

or

```bash
uv run python mlp_numpy.py
```

---

## 📊 Expected Output

* Training loss printed every 100 epochs
* Final test accuracy
* Decision boundary visualization
* Training loss graph

Example:

```text
Epoch 0, Loss: 0.6931
Epoch 100, Loss: 0.5124
Epoch 200, Loss: 0.3918
...
Test Accuracy: 0.9500
```

---

## 🧠 Neural Network Architecture

```text
Input Layer (2 Features)
        │
        ▼
Hidden Layer (10 Neurons)
  Sigmoid Activation
        │
        ▼
Output Layer (1 Neuron)
  Sigmoid Activation
        │
        ▼
Binary Prediction (0 or 1)
```

---

## 🔄 Project Workflow

```text
Generate Dataset
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Initialize MLP
        │
        ▼
Forward Propagation
        │
        ▼
Loss Calculation
        │
        ▼
Backpropagation
        │
        ▼
Update Weights
        │
        ▼
Repeat for 1000 Epochs
        │
        ▼
Model Evaluation
        │
        ▼
Decision Boundary & Loss Visualization
```

---

## 📚 Learning Outcomes

After completing this project, you will understand:

* Neural Network Fundamentals
* Multi-Layer Perceptron (MLP)
* Matrix Operations in Neural Networks
* Weight and Bias Initialization
* Sigmoid Activation Function
* Binary Cross-Entropy Loss
* Forward Propagation
* Backpropagation
* Gradient Descent Optimization
* Data Preprocessing
* Model Evaluation
* Decision Boundary Visualization

---

## 🎯 Future Improvements

* ReLU and Tanh activation functions
* Multiple hidden layers
* Softmax for multi-class classification
* Mini-batch Gradient Descent
* Adam Optimizer
* Early Stopping
* Model Saving and Loading
* Support for custom datasets

---

## 📜 License

This project is released under the MIT License.

---

## 👨‍💻 Author

**Pandeeswaran P**

* B.Tech Artificial Intelligence and Data Science
* Passionate about Artificial Intelligence, Machine Learning, Deep Learning, and Large Language Models (LLMs)

⭐ If you found this project useful, consider giving it a star on GitHub!
