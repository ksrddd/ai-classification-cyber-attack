from __future__ import annotations

from src.training.checkpoints import checkpoint_matches, load_checkpoint, write_checkpoint


def test_checkpoint_round_trip_is_atomic_and_bound_to_signature(tmp_path) -> None:
    path = tmp_path / "checkpoints" / "xgboost.json"
    write_checkpoint(
        path,
        {"model": "xgboost", "run_signature": "abc", "phase": "model_ready"},
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint is not None
    assert checkpoint["phase"] == "model_ready"
    assert checkpoint_matches(checkpoint, model_name="xgboost", run_signature="abc")
    assert not checkpoint_matches(checkpoint, model_name="xgboost", run_signature="other")


def test_invalid_checkpoint_is_not_reused(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text("not json", encoding="utf-8")
    assert load_checkpoint(path) is None
