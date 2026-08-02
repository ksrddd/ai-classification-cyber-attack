"""The seven classifiers compared on CSE-CIC-IDS2018.

Each builder returns a bare scikit-learn-compatible estimator. Scaling is
already handled by :class:`~src.ids2018.preprocessing.Ids2018Preprocessor`
before anything reaches these models, so no Pipeline wrapping is needed --
which also keeps the saved artefacts small.

Class imbalance is severe (Benign is ~83% of the corpus, SQL Injection is
0.0005%), but it is handled by one global switch -- see
:func:`set_class_weighting` -- rather than per model. Weighting only the
models that support it would break the comparison: ``MLPClassifier`` accepts
neither ``class_weight`` nor ``sample_weight``, so it would be the only
model free to favour the majority class, and would top the accuracy column
for that reason alone.

``accelerator="gpu"`` moves XGBoost and CatBoost (and the XGBoost base
learner inside the stacking ensemble) onto CUDA. Nothing else can follow:
scikit-learn has no CUDA backend, and the LightGBM wheel published on PyPI
is built without GPU support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.ids2018.config import CLASS_WEIGHT_PARAMS, MODEL_PARAMS, RANDOM_STATE

logger = logging.getLogger(__name__)

# Set once per run by :func:`set_class_weighting`. A module-level switch
# keeps the builder signatures uniform -- every builder takes the same two
# device arguments and nothing else, so BUILDERS stays a plain lookup.
_CLASS_WEIGHTING = "none"


def set_class_weighting(mode: str) -> None:
    """Select how every model handles class imbalance, for the whole run."""
    if mode not in {"none", "balanced"}:
        raise ValueError(f"Unknown class weighting {mode!r}; use 'none' or 'balanced'")
    global _CLASS_WEIGHTING
    _CLASS_WEIGHTING = mode


def _params(name: str, **overrides: Any) -> dict[str, Any]:
    """Hyper-parameters for ``name``, with class weighting applied if enabled."""
    params = dict(MODEL_PARAMS[name])
    if _CLASS_WEIGHTING == "balanced":
        params.update(CLASS_WEIGHT_PARAMS.get(name, {}))
    params.update(overrides)
    return params


# Models that can actually use ``--accelerator gpu``. Everything else stays
# on the CPU no matter what is passed.
GPU_CAPABLE = frozenset({"xgboost", "catboost", "stacking"})


# ----------------------------------------------------------------------
# Device placement
# ----------------------------------------------------------------------
def _xgb_device(accelerator: str, gpu_devices: str) -> dict[str, Any]:
    """XGBoost 2.x selects the device with ``device=``, not ``tree_method``."""
    if accelerator != "gpu":
        return {"device": "cpu"}
    first = gpu_devices.split(",")[0].strip()
    return {"device": f"cuda:{first}"}


def _catboost_device(accelerator: str, gpu_devices: str) -> dict[str, Any]:
    if accelerator != "gpu":
        return {"task_type": "CPU"}
    return {"task_type": "GPU", "devices": gpu_devices}


# ----------------------------------------------------------------------
# Individual builders
# ----------------------------------------------------------------------
def build_random_forest(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Bagged trees. Strong, fast baseline and the usual NIDS reference. CPU only."""
    return RandomForestClassifier(**_params("random_forest"))


def build_xgboost(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Histogram gradient boosting. ``hist`` runs on both CPU and CUDA."""
    return XGBClassifier(**_params("xgboost"), **_xgb_device(accelerator, gpu_devices))


def build_lightgbm(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Leaf-wise gradient boosting; typically the fastest of the three GBDTs.

    Stays on the CPU: the PyPI wheel is built without the OpenCL/CUDA
    backend, so requesting a GPU device here would only raise.
    """
    from lightgbm import LGBMClassifier

    if accelerator == "gpu":
        logger.info("  lightgbm has no GPU support in the PyPI wheel -- running on CPU")
    return LGBMClassifier(**_params("lightgbm"))


def build_catboost(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Ordered boosting with symmetric trees; least sensitive to defaults."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        **_params("catboost"), **_catboost_device(accelerator, gpu_devices)
    )


def build_mlp(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Two-layer perceptron. Requires the scaled features it is given. CPU only."""
    return MLPClassifier(**_params("mlp"))


def build_logistic_regression(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Linear baseline -- how much of the task is linearly separable. CPU only."""
    return LogisticRegression(**_params("logistic_regression"))


class LabelSafeXGBClassifier(XGBClassifier):
    """XGBClassifier that tolerates gaps in the label space.

    ``StackingClassifier`` refits every base learner on CV folds. A class
    with only a handful of training rows -- SQL Injection has exactly one --
    is absent from most folds, so the fold's ``y`` skips a code (0..12, 14).
    XGBoost rejects non-contiguous labels outright, which kills the whole
    ensemble; RandomForest and LightGBM accept them without complaint.

    This subclass re-encodes ``y`` to a dense range for fitting, then reports
    the *original* labels in ``classes_``. scikit-learn's
    ``cross_val_predict`` reads that attribute and pads the missing
    probability columns itself, so the meta-learner always sees one column
    per class, whichever fold produced the row.

    Defined at module level rather than inside a factory so that a fitted
    ensemble containing it can be pickled -- joblib resolves classes by
    qualified name, and a class closed over by a function has none it can
    find.
    """

    # XGBoost defines classes_ as a read-only property returning
    # np.arange(n_classes_), so it must be overridden, not assigned to.
    # Timing matters: fit() sets n_classes_ and then validates np.unique(y)
    # against self.classes_, so during that check the property has to keep
    # returning the dense range. Only once fitting is done may it report
    # the real codes.
    @property
    def classes_(self) -> np.ndarray:
        present = getattr(self, "_present_classes", None)
        return np.arange(self.n_classes_) if present is None else present

    def fit(self, X, y, **kwargs):
        self._present_classes = None
        present = np.unique(y)
        super().fit(X, np.searchsorted(present, y), **kwargs)
        self._present_classes = present
        return self

    def predict(self, X, **kwargs):
        # XGBoost's predict returns argmax over the dense columns, so the
        # result is an index into _present_classes, not a label.
        return self._present_classes[super().predict(X, **kwargs)]


def build_stacking(accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Stacking ensemble: RF + LightGBM + XGBoost -> logistic meta-learner.

    Base learners are deliberately lighter than their standalone versions:
    ``StackingClassifier`` refits each of them ``cv + 1`` times, so a full
    400-tree configuration would cost ~4x the standalone run for a marginal
    gain. The meta-learner consumes out-of-fold class probabilities.
    """
    from lightgbm import LGBMClassifier

    cfg = MODEL_PARAMS["stacking"]

    # Base learners never carry class_weight of their own: when weighting is
    # enabled the ensemble is fitted with balanced *sample* weights (see
    # fit_model), which StackingClassifier forwards down. Setting both would
    # weight the rare classes twice.
    rf_params = {**MODEL_PARAMS["random_forest"], "n_estimators": 150, "max_depth": 20}
    lgb_params = {**MODEL_PARAMS["lightgbm"], "n_estimators": 200}
    # Only the XGBoost base learner can move to CUDA; RF and LightGBM cannot.
    xgb_params = {
        **MODEL_PARAMS["xgboost"],
        "n_estimators": 200,
        "max_depth": 6,
        **_xgb_device(accelerator, gpu_devices),
    }

    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(**rf_params)),
            ("lgbm", LGBMClassifier(**lgb_params)),
            ("xgb", LabelSafeXGBClassifier(**xgb_params)),
        ],
        final_estimator=LogisticRegression(
            max_iter=cfg["final_estimator_max_iter"],
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=cfg["cv"],
        n_jobs=cfg["n_jobs"],
        passthrough=cfg["passthrough"],
    )


BUILDERS: dict[str, Any] = {
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
    "mlp": build_mlp,
    "logistic_regression": build_logistic_regression,
    "stacking": build_stacking,
}

# Models with no ``class_weight`` parameter; they get sample weights instead.
_NEEDS_SAMPLE_WEIGHT = {"xgboost", "stacking"}


def build_model(name: str, accelerator: str = "cpu", gpu_devices: str = "0") -> BaseEstimator:
    """Instantiate one model by name, placed on the requested device."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(sorted(BUILDERS))}")
    if accelerator == "gpu" and name not in GPU_CAPABLE:
        logger.info("  %s is CPU-only -- --accelerator gpu does not apply", name)
    return BUILDERS[name](accelerator, gpu_devices)


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_model(name: str, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit ``model``, applying balanced sample weights where needed.

    ``StackingClassifier`` forwards ``sample_weight`` to every base learner
    that accepts one, so a single call covers the whole ensemble.
    """
    if _CLASS_WEIGHTING == "balanced" and name in _NEEDS_SAMPLE_WEIGHT:
        weights = compute_sample_weight("balanced", y)
        logger.info("  applying balanced sample weights (no class_weight support)")
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_model(name: str, model: BaseEstimator, models_dir: Path) -> list[Path]:
    """Save a fitted model, preferring each library's native format.

    Native formats (XGBoost ``.json``, CatBoost ``.cbm``) survive library
    upgrades, whereas a pickled estimator does not. The joblib copy is
    still written for every model because it is what the evaluation and
    inference code loads directly.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    joblib_path = models_dir / f"{name}.joblib"
    joblib.dump(model, joblib_path, compress=3)
    written.append(joblib_path)

    if name == "xgboost":
        native = models_dir / f"{name}.json"
        model.get_booster().save_model(str(native))
        written.append(native)
    elif name == "catboost":
        native = models_dir / f"{name}.cbm"
        model.save_model(str(native))
        written.append(native)
    elif name == "lightgbm":
        native = models_dir / f"{name}.txt"
        model.booster_.save_model(str(native))
        written.append(native)

    logger.info("  saved -> %s", ", ".join(p.name for p in written))
    return written
