"""The screening threshold's fitting rule, pinned on synthetic records.

scripts/evaluation/06_screening_threshold.py fits a decision threshold on each
fold's inner-validation records and only then applies it to test. A threshold
moved off 0.5 is a fitted statistic (split rule 5), so the rule that fits it
deserves its own guard: it must reach the target sensitivity on the records it
was fitted on, sit between observed scores rather than exactly on one, and
never rise when more sensitivity is asked for.
"""

import importlib.util
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "evaluation", "06_screening_threshold.py")


def load_module():
    spec = importlib.util.spec_from_file_location("screening_threshold", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_records():
    """10 abnormal and 10 normal records with overlapping scores."""
    abnormal = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]
    normal = [0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]
    return pd.DataFrame(
        {
            "record_id": ["r" + str(i) for i in range(20)],
            "label": [1] * 10 + [0] * 10,
            "prob": abnormal + normal,
        }
    )


@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 1.0])
def test_threshold_reaches_target_on_the_records_it_was_fitted_on(target):
    module = load_module()
    records = make_records()
    threshold = module.choose_threshold(records, target)
    sensitivity = module.rates(records, threshold)[0]
    assert sensitivity >= target, f"target {target}: fitted threshold {threshold} gives {sensitivity}"


def test_threshold_sits_midway_between_observed_scores():
    """Nine of ten abnormal records reach 0.3; the next lower score is 0.25, so
    the threshold belongs at 0.275, not on either record."""
    module = load_module()
    assert module.choose_threshold(make_records(), 0.9) == pytest.approx(0.275)


def test_asking_for_more_sensitivity_never_raises_the_threshold():
    module = load_module()
    records = make_records()
    thresholds = [module.choose_threshold(records, target) for target in [0.5, 0.7, 0.9, 1.0]]
    for index in range(len(thresholds) - 1):
        assert thresholds[index] >= thresholds[index + 1], f"thresholds not monotone: {thresholds}"


def test_threshold_is_fitted_on_validation_never_on_test():
    """Give validation and test records different scores: the fitted threshold
    must be the one validation alone implies, whatever test looks like."""
    module = load_module()
    val = make_records()
    val["fold"] = 0
    val["split"] = "val"
    test = make_records()
    test["prob"] = test["prob"] * 0.5
    test["fold"] = 0
    test["split"] = "test"
    predictions = pd.concat([val, test], ignore_index=True)
    result = module.evaluate_target(predictions, 0.9, 1)
    assert result["threshold"].iloc[0] == pytest.approx(module.choose_threshold(val, 0.9))
    assert result["threshold"].iloc[0] != pytest.approx(module.choose_threshold(test, 0.9))


def test_default_threshold_is_the_neutral_one():
    assert load_module().DEFAULT_THRESHOLD == 0.5
