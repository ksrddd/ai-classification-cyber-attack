"""Cyber Attack Classification — top-level source package.

Modules
-------
config         Project-wide constants and YAML config loader.
data           CSV loading, schema validation, EDA helpers.
features       Cleaning, encoding, feature selection, sklearn Pipeline.
models         Logistic Regression, Random Forest, MLP wrappers + GridSearch.
evaluation     Metric computation, confusion matrix, comparison reports.
explainability SHAP analyzer.
visualization  Plot helpers.
utils          Logging, I/O, deterministic seeding.
pipelines      End-to-end stage runners called by main.py.
"""
__version__ = "0.1.0"
