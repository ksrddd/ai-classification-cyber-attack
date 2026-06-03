"""Model wrappers + GridSearchCV tuner.

Each model is a thin class around the underlying sklearn estimator that:
- Reads hyperparameters from ``config.yaml::models.<name>``.
- Builds itself wrapped in the standard Pipeline (scaler + clf).
- Exposes ``fit``, ``predict``, ``predict_proba``, ``save``, ``load``.
- Uses ``RANDOM_STATE`` for any stochastic component.
"""
