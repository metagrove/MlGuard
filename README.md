# MLGuard

![PyPI](https://img.shields.io/pypi/v/mlguardlabs)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**MLGuard** is a lightweight **machine learning experiment management and data validation toolkit** designed to help data scientists quickly validate datasets, detect potential data leakage, and benchmark multiple machine learning models.

It provides a simple interface to run experiments and compare models with minimal setup.

---

# Why MLGuard?

When working with machine learning pipelines, developers often face problems such as:

* Undetected **data leakage**
* Repeated **model experimentation scripts**
* Difficulty comparing models quickly
* Poor experiment organization

MLGuard helps solve these problems by providing:

* Automated **data leakage detection**
* Automatic **problem type detection**
* Built-in **experiment manager**
* Simple **model comparison tools**

---

# Features

### Data Leakage Detection

Detects potential **target leakage** by analyzing correlations between features and the target variable.

### Experiment Manager

Runs multiple machine learning models automatically and evaluates their performance.

### Model Comparison

Compares model performance using appropriate evaluation metrics.

### Automatic Problem Detection

Detects whether the task is:

* **Regression**
* **Classification**

---

# Installation

Install from PyPI:

```bash
pip install mlguardlabs
```

Install locally for development:

```bash
pip install -e .
```

---

# Quick Example

## Regression Example

```python
from sklearn.datasets import fetch_california_housing
from mlguard import ExperimentManager

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

# Classification Example

```python
from sklearn.datasets import load_iris
from mlguard import ExperimentManager

data = load_iris(as_frame=True)

X = data.data
y = data.target

exp = ExperimentManager()

exp.fit(X, y)

print(exp.compare())
```

---

# Data Leakage Detection Example

```python
from mlguard import LeakageDetector

detector = LeakageDetector()

detector.fit(X, y)

print(detector.report())
```

This prints features that have unusually high correlation with the target.

---

# How MLGuard Works

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

This allows users to quickly answer:

**Which model works best for my dataset?**

---

# Project Structure

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

# Example Workflow

Typical machine learning workflow using MLGuard:

```
EDA
↓
Feature Engineering
↓
Feature Selection
↓
MLGuard Experiment Manager
↓
Best Model Selection
↓
Hyperparameter Tuning
↓
Final Model
```

MLGuard helps simplify the **experimentation stage**.

---

# Roadmap

Future planned features include:

* Cross-validation experiment engine
* Dataset inspector
* Automatic preprocessing pipelines
* Feature validation tools
* Experiment tracking

---

# Contributing

Contributions are welcome!

Steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

# License

MIT License © 2026 Tarun M
