# MLGuard

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**MLGuard** is a lightweight machine learning **experiment management and data validation toolkit** designed to help detect potential data leakage and benchmark multiple models quickly.

---

## 🚀 Features

* 🔍 **Data Leakage Detection**
  Detects potential target leakage by analyzing correlations between features and the target.

* ⚙️ **Experiment Manager**
  Automatically runs multiple ML models and evaluates them.

* 📊 **Model Comparison**
  Compare model performance using appropriate metrics.

* 🧠 **Automatic Problem Detection**
  Detects whether the task is **regression or classification**.

---

## 📦 Installation

```bash
pip install mlguard
```

or install locally:

```bash
pip install -e .
```

---

## ⚡ Quick Example

### Regression Example

```python
from sklearn.datasets import fetch_california_housing
from mlguard.experiments import ExperimentManager

data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

exp = ExperimentManager()

exp.fit(X, y)

print(exp.compare())
```

Example output:

```
Running leakage detection...
No leakage detected

Detected problem type: regression

[('rf', 0.50), ('ridge', 0.74)]
```

---

### Classification Example

```python
from sklearn.datasets import load_iris
from mlguard.experiments import ExperimentManager

data = load_iris(as_frame=True)

X = data.data
y = data.target

exp = ExperimentManager()

exp.fit(X, y)

print(exp.compare())
```

---

## 🏗 Project Structure

```
mlguard/
│
├── experiments
│   └── experiment_manager.py
│
├── inspection
│   └── leakage_detector.py
│
├── metrics
│   ├── regression_metrics.py
│   └── classification_metrics.py
│
├── utils
│   └── problem_type.py
│
├── examples
├── tests
```

---

## 🧠 How MLGuard Works

MLGuard runs a lightweight validation and experimentation pipeline:

```
Dataset
   ↓
Leakage Detection
   ↓
Problem Type Detection
   ↓
Model Experiments
   ↓
Model Evaluation
   ↓
Model Comparison
```

This helps ML practitioners quickly answer:

> **Which model works best for my dataset?**

---

## 🛣 Roadmap

Planned improvements:

* Cross-validation experiment engine
* Dataset inspector
* Automatic preprocessing pipelines
* Feature validation tools
* Experiment tracking

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

## 📜 License

MIT License © 2026 Tarun
