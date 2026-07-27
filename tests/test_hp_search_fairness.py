"""Tests for the hyperparameter-search fairness contract.

The evaluation standard requires every model to be tuned by the same method,
with the same budget. These tests pin the properties that make that claim true,
and guard the two ways it has already been broken in practice.
"""

from __future__ import annotations

import pytest
from sklearn.model_selection import ParameterGrid

import train

TUNABLE = [
    name for name in train.CONFIG["models"] if train.hp_grids(name)
]


def test_every_tunable_model_has_the_same_search_space_size() -> None:
    """Equal trial counts over unequal spaces is not equal treatment.

    Before this was enforced, logistic regression had 8 combinations (fully
    enumerated after 8 draws) while XGBoost had 256 (8% explored at n_iter=20).
    """
    sizes = {name: train.hp_search_space_size(name) for name in TUNABLE}

    assert set(sizes.values()) == {train.HP_SEARCH_SPACE_SIZE}, sizes


def test_search_space_size_matches_the_actual_grid() -> None:
    for name in TUNABLE:
        assert train.hp_search_space_size(name) == len(
            list(ParameterGrid(train.hp_grids(name)))
        ), name


def test_untuned_models_are_declared_not_silent() -> None:
    untuned = [n for n in train.CONFIG["models"] if not train.hp_grids(n)]

    assert set(untuned) == set(train.HP_UNTUNED_MODELS)
    for reason in train.HP_UNTUNED_MODELS.values():
        assert reason.strip()


def test_n_iter_never_exceeds_the_search_space() -> None:
    """RandomizedSearchCV warns when asked for more draws than combinations.

    Under this project's ``-W error::Warning`` clean run that warning is an
    exception, so every preset must stay within the space.
    """
    for preset_name, preset in train.RAM_PRESETS.items():
        n_iter = preset["hp_search_n_iter"]
        assert n_iter <= train.HP_SEARCH_SPACE_SIZE, (preset_name, n_iter)


def test_no_grid_tunes_the_optimiser_budget() -> None:
    """``max_iter``/``n_estimators``-style budgets are fine; iteration ceilings
    that can sit *below* the configured budget are not.

    Tuning ``max_iter`` made the search compare under-converged fits against
    converged ones and emitted ConvergenceWarning. It is a budget, not a
    property of the model.
    """
    for name in TUNABLE:
        assert not any("max_iter" in key for key in train.hp_grids(name)), name


def test_no_grid_overrides_class_weight() -> None:
    """Imbalance handling is set by --imbalance-strategy for every model.

    A grid that tuned class_weight would let one model opt out of the shared
    treatment, so it would stop being a controlled variable.
    """
    for name in TUNABLE:
        assert not any("class_weight" in key for key in train.hp_grids(name)), name


def test_logistic_regression_tunes_only_real_hyperparameters() -> None:
    grid = train.hp_grids("logistic_regression")

    assert set(grid) == {"clf__C", "clf__tol", "clf__fit_intercept"}
    # Every tolerance is at or above the sklearn default, so none of them
    # needs more iterations than the configured budget already allows.
    assert min(grid["clf__tol"]) >= 1e-4


def test_logistic_regression_budget_exceeds_observed_need() -> None:
    """Regression guard for the ConvergenceWarning.

    Measured on the 200k-row HP subset of the real corpus, lbfgs needs 523
    iterations at tol=1e-4 and 84 at tol=1e-3. The configured budget must stay
    comfortably above the worst case.
    """
    clf = train.build_pipeline("logistic_regression", 9, 42).named_steps["clf"]

    assert clf.max_iter >= 1000


# ---------------------------------------------------------------------------
# Grid identity
# ---------------------------------------------------------------------------
def test_grid_fingerprint_is_stable_and_distinct() -> None:
    assert train.hp_grid_fingerprint("logistic_regression") == \
        train.hp_grid_fingerprint("logistic_regression")
    assert train.hp_grid_fingerprint("logistic_regression") != \
        train.hp_grid_fingerprint("random_forest")


def test_editing_a_grid_invalidates_the_saved_model(monkeypatch) -> None:
    """Space size alone cannot detect changed *values*."""
    cfg = {**train.CONFIG, "data_fingerprint": "abc", "hp_search": True}
    saved = train._imbalance_metadata(cfg, "logistic_regression")
    assert train._imbalance_config_matches(saved, cfg, "logistic_regression")

    original = train.hp_grids

    def edited(model_name: str):
        grid = dict(original(model_name))
        if model_name == "logistic_regression":
            grid["clf__C"] = [c * 2 for c in grid["clf__C"]]
        return grid

    monkeypatch.setattr(train, "hp_grids", edited)

    assert not train._imbalance_config_matches(saved, cfg, "logistic_regression")


def test_artifacts_predating_the_fingerprint_are_still_reusable() -> None:
    """Adding this key must not force a wholesale retrain of finished work."""
    cfg = {**train.CONFIG, "data_fingerprint": "abc", "hp_search": True}
    saved = train._imbalance_metadata(cfg, "random_forest")
    saved.pop("hp_grid_fingerprint")

    assert train._imbalance_config_matches(saved, cfg, "random_forest")


@pytest.mark.parametrize("model_name", TUNABLE)
def test_grid_parameters_are_accepted_by_the_estimator(model_name) -> None:
    """A typo in a grid key only fails deep inside the search otherwise.

    Checked with ``set_params`` rather than membership in ``get_params``:
    CatBoost reports only the parameters that were explicitly passed to it, so
    a perfectly valid knob like ``l2_leaf_reg`` is absent from ``get_params``
    until it is set.
    """
    pipeline = train.build_pipeline(model_name, 9, 42)
    first_of_each = {
        key: values[0] for key, values in train.hp_grids(model_name).items()
    }

    pipeline.set_params(**first_of_each)
