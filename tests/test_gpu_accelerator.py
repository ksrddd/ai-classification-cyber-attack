"""Tests for GPU selection and its effect on artifact reuse.

``--accelerator gpu`` is a run-level request that only three models act on.
Getting that wrong is expensive in both directions: too strict and a CPU->GPU
switch discards hours of finished CPU-only work; too loose and a CPU-trained
model is silently reported as a GPU result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train


def _cfg(**overrides) -> dict:
    return {**train.CONFIG, **overrides}


# ---------------------------------------------------------------------------
# Which models the accelerator actually reaches
# ---------------------------------------------------------------------------
def test_gpu_capable_set_matches_what_build_pipeline_does() -> None:
    """The constant must track reality, or reuse decisions go wrong.

    A model belongs in GPU_CAPABLE_MODELS exactly when its pipeline differs
    between cpu and gpu. This catches the set drifting away from the builder.
    """
    differs = set()
    for name in train.CONFIG["models"]:
        cpu = train.build_pipeline(name, 9, 42, accelerator="cpu")
        gpu = train.build_pipeline(name, 9, 42, accelerator="gpu")
        cpu_params, gpu_params = cpu.get_params(), gpu.get_params()
        if any(
            repr(value) != repr(gpu_params.get(key))
            for key, value in cpu_params.items()
            if key in gpu_params
        ):
            differs.add(name)

    assert differs == set(train.GPU_CAPABLE_MODELS)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("xgboost", "gpu"),
        ("catboost", "gpu"),
        ("stacking", "gpu"),
        ("random_forest", "cpu"),
        ("lightgbm", "cpu"),
        ("mlp", "cpu"),
        ("logistic_regression", "cpu"),
    ],
)
def test_effective_accelerator_is_per_model(model_name, expected) -> None:
    assert train.effective_accelerator(_cfg(accelerator="gpu"), model_name) == expected


def test_effective_accelerator_without_a_model_is_the_run_level_request() -> None:
    assert train.effective_accelerator(_cfg(accelerator="gpu")) == "gpu"
    assert train.effective_accelerator(_cfg(accelerator="cpu")) == "cpu"


def test_cpu_run_keeps_every_model_on_cpu() -> None:
    cfg = _cfg(accelerator="cpu")
    for name in train.CONFIG["models"]:
        assert train.effective_accelerator(cfg, name) == "cpu"


# ---------------------------------------------------------------------------
# Artifact reuse across an accelerator switch
# ---------------------------------------------------------------------------
@pytest.fixture
def saved_cpu_artifact() -> dict:
    """Metrics as written by a completed CPU run."""
    cfg = _cfg(accelerator="cpu", gpu_devices="0", data_fingerprint="abc")
    return train._imbalance_metadata(cfg, "random_forest")


def test_cpu_only_model_is_reused_when_the_run_switches_to_gpu(
    saved_cpu_artifact,
) -> None:
    """The whole point: GPU cannot change a RandomForest, so keep the artifact."""
    gpu_cfg = _cfg(accelerator="gpu", gpu_devices="0", data_fingerprint="abc")

    assert train._imbalance_config_matches(
        saved_cpu_artifact, gpu_cfg, "random_forest"
    )


def test_gpu_capable_model_is_retrained_when_the_run_switches_to_gpu() -> None:
    cfg_cpu = _cfg(accelerator="cpu", gpu_devices="0", data_fingerprint="abc")
    saved = train._imbalance_metadata(cfg_cpu, "xgboost")
    gpu_cfg = _cfg(accelerator="gpu", gpu_devices="0", data_fingerprint="abc")

    assert not train._imbalance_config_matches(saved, gpu_cfg, "xgboost")


def test_gpu_device_change_only_invalidates_gpu_models() -> None:
    gpu_cfg = _cfg(accelerator="gpu", gpu_devices="0", data_fingerprint="abc")
    other_device = _cfg(accelerator="gpu", gpu_devices="1", data_fingerprint="abc")

    rf_saved = train._imbalance_metadata(gpu_cfg, "random_forest")
    xgb_saved = train._imbalance_metadata(gpu_cfg, "xgboost")

    assert train._imbalance_config_matches(rf_saved, other_device, "random_forest")
    assert not train._imbalance_config_matches(xgb_saved, other_device, "xgboost")


def test_legacy_artifact_without_requested_accelerator_still_matches() -> None:
    """Artifacts written before this distinction carry no new key."""
    cfg = _cfg(accelerator="cpu", gpu_devices="0", data_fingerprint="abc")
    legacy = train._imbalance_metadata(cfg, "lightgbm")
    legacy.pop("requested_accelerator")

    gpu_cfg = _cfg(accelerator="gpu", gpu_devices="0", data_fingerprint="abc")
    assert train._imbalance_config_matches(legacy, gpu_cfg, "lightgbm")


def test_metadata_records_both_effective_and_requested() -> None:
    cfg = _cfg(accelerator="gpu")

    rf = train._imbalance_metadata(cfg, "random_forest")
    xgb = train._imbalance_metadata(cfg, "xgboost")

    assert rf["accelerator"] == "cpu"          # honest: it did not use the GPU
    assert rf["requested_accelerator"] == "gpu"
    assert xgb["accelerator"] == "gpu"
    assert xgb["requested_accelerator"] == "gpu"


def test_checkpoint_signature_separates_cpu_and_gpu_only_where_it_matters() -> None:
    cpu_cfg = _cfg(accelerator="cpu", data_fingerprint="abc")
    gpu_cfg = _cfg(accelerator="gpu", data_fingerprint="abc")

    assert train._checkpoint_signature(cpu_cfg, "random_forest") == \
        train._checkpoint_signature(gpu_cfg, "random_forest")
    assert train._checkpoint_signature(cpu_cfg, "xgboost") != \
        train._checkpoint_signature(gpu_cfg, "xgboost")


# ---------------------------------------------------------------------------
# Latency must be measured on one device for every model
# ---------------------------------------------------------------------------
def test_inference_context_moves_cuda_estimators_to_cpu_and_restores() -> None:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(device="cuda", n_estimators=2)),
    ])

    with train._inference_on_cpu(pipeline) as moved:
        assert moved is True
        assert pipeline.get_params()["clf__device"] == "cpu"

    assert pipeline.get_params()["clf__device"] == "cuda"


def test_inference_context_is_a_no_op_for_cpu_models() -> None:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(device="cpu", n_estimators=2)),
    ])

    with train._inference_on_cpu(pipeline) as moved:
        assert moved is False
        assert pipeline.get_params()["clf__device"] == "cpu"


def test_measure_inference_cost_reports_where_it_timed() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = (rng.random(200) > 0.5).astype(int)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=3, max_depth=2, device="cpu")),
    ]).fit(X, y)

    cost = train.measure_inference_cost(pipeline, X, batch_size=50, repeats=3)

    assert cost["predict_device"] == "cpu"
    assert cost["predict_moved_from_gpu"] is False
    assert cost["predict_latency_p95_ms"] >= cost["predict_latency_p50_ms"]
    assert cost["throughput_flows_per_sec"] > 0
    assert set(cost) <= set(train.EFFICIENCY_KEYS)


def test_measure_inference_cost_handles_an_empty_test_set() -> None:
    pipeline = Pipeline([("scaler", StandardScaler())])
    assert train.measure_inference_cost(pipeline, pd.DataFrame()) == {}
