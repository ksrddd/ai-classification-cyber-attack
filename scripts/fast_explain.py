"""fast_explain.py

รัน SHAP explain โดย inject BalancedXGBClassifier / _LGBMNoFeatureNamesCheck
เข้า __main__ ก่อน ซึ่งแก้ปัญหา pickle ที่ train.py save โมเดลไว้ใน __main__
แต่ main.py ไม่มี class เหล่านี้

หลักการ: joblib.load() ใช้ pickle ซึ่ง resolve class จาก module ที่ save ไว้
เมื่อ train.py เป็น __main__ class ถูก serialize เป็น '__main__.BalancedXGBClassifier'
ดังนั้นตอน load ต้องมี class นั้นอยู่ใน sys.modules['__main__']
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from pathlib import Path

from lightgbm import LGBMClassifier

# --- inject BalancedXGBClassifier into __main__ BEFORE any joblib.load ---
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import __main__


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier with auto balanced sample_weight (mirrors train.py)."""
    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


class _LGBMNoFeatureNamesCheck(LGBMClassifier):
    """LightGBM that silences feature_names_in_ warning (mirrors train.py)."""
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        if hasattr(self, "feature_names_in_"):
            with contextlib.suppress(AttributeError):
                object.__delattr__(self, "feature_names_in_")
        return self


# Inject into __main__ so pickle.load() can resolve them
__main__.BalancedXGBClassifier   = BalancedXGBClassifier
__main__._LGBMNoFeatureNamesCheck = _LGBMNoFeatureNamesCheck

# Also register under src.models.xgboost_model so any code using that path works
xgb_mod = importlib.import_module("src.models.xgboost_model")
xgb_mod.BalancedXGBClassifier = BalancedXGBClassifier

# --- Now run the explain pipeline ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.config.constants import CONFIG_PATH  # noqa: E402
from src.pipelines.explain import run as run_explain  # noqa: E402

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast SHAP explain with pickle shim")
    parser.add_argument("--model", default="all",
                        help="Model name or 'all' (default: all)")
    args = parser.parse_args()

    logging.getLogger(__name__).info(
        "Running explain for model=%s (pickle shim active)", args.model
    )
    result = run_explain(CONFIG_PATH, model=args.model)
    logging.getLogger(__name__).info("Done! SHAP page should now load correctly.")
    for name, info in result.get("models", {}).items():
        logging.getLogger(__name__).info(
            "  %s: explainer=%s, top_feature=%s",
            name, info["explainer"],
            info["top_overall"][0][0] if info["top_overall"] else "—",
        )
