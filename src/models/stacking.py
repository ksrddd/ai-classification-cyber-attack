"""Stacking ensemble: LightGBM + XGBoost + RandomForest -> LogisticRegression.

Uses sklearn's StackingClassifier. The three base learners use histogram-based
tree methods (fast on CICIDS2017's 300K rows). The meta-learner is a tuned
LogisticRegression trained on the out-of-fold predictions.

Why this configuration:
- LightGBM + XGBoost: complementary gradient boosting flavours; different
  regularisation paths often catch different minorities.
- RandomForest: high-variance bagged trees balance the low-variance boosters.
- LogisticRegression meta: linear blend is calibrated and interpretable;
  class_weight=balanced keeps Bot/Web Attack from being ignored.
"""

from __future__ import annotations

import logging

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.constants import RANDOM_STATE
from src.models.base import BaseModel
from src.models.xgboost_model import BalancedXGBClassifier

logger = logging.getLogger(__name__)


class StackingModel(BaseModel):
    name = "stacking"

    def _build_estimator(self) -> StackingClassifier:
        p = self.baseline_params

        base_estimators = [
            ("lgbm", LGBMClassifier(
                n_estimators=p.get("lgbm_n_estimators", 300),
                num_leaves=p.get("lgbm_num_leaves", 63),
                learning_rate=p.get("lgbm_lr", 0.05),
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=5,
                class_weight="balanced",
                n_jobs=-1,
                verbosity=-1,
                random_state=RANDOM_STATE,
            )),
            ("xgb", BalancedXGBClassifier(
                n_estimators=p.get("xgb_n_estimators", 200),
                max_depth=p.get("xgb_max_depth", 8),
                learning_rate=p.get("xgb_lr", 0.1),
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1,
                verbosity=0,
                random_state=RANDOM_STATE,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=p.get("rf_n_estimators", 200),
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]

        meta = LogisticRegression(
            C=p.get("meta_C", 1.0),
            solver="lbfgs",
            max_iter=p.get("meta_max_iter", 1000),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        # n_jobs=1 here: each base learner already uses all cores internally
        # (n_jobs=-1). Running them in parallel on top would oversubscribe
        # the CPU and make the machine unresponsive.
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=p.get("cv", 5),
            stack_method="predict_proba",
            n_jobs=1,
            passthrough=False,
        )

    def build(self) -> Pipeline:
        """Wrap StackingClassifier in a standard scaler pipeline.

        The outer StandardScaler is shared by all base estimators. Tree models
        are invariant to scaling, and the meta-learner (LogisticRegression)
        benefits from it.
        """
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        stacker = self._build_estimator()
        self.pipeline = Pipeline([
            ("scaler", scaler),
            ("clf",    stacker),
        ])
        logger.debug("Built Stacking pipeline: scaler -> (lgbm + xgb + rf) -> logreg")
        return self.pipeline
