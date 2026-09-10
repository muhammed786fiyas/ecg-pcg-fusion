"""assign_folds must be structurally incapable of seeing segment-derived data.

The rule is not "we were careful". It is that the function's signature accepts
only record IDs, record-level labels and configuration scalars, so there is
nowhere for a segment ID or a signal to enter. These tests assert that by
inspecting the signature and by feeding the function data it must not be able
to exploit.
"""

import importlib.util
import inspect
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "data", "05_assign_folds.py")

ALLOWED_PARAMETERS = {
    "record_ids",
    "labels",
    "seed",
    "k",
    "dev_train_frac",
    "dev_val_frac",
    "dev_test_frac",
    "cv_inner_val_frac",
}

FORBIDDEN_SUBSTRINGS = ["segment", "signal", "ecg", "pcg", "scalogram", "row_index", "variant"]


def load_module():
    spec = importlib.util.spec_from_file_location("assign_folds", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assign_folds_signature_admits_only_records_labels_and_config():
    """The structural guarantee. If someone adds a segments= argument, this fails."""
    module = load_module()
    parameters = set(inspect.signature(module.assign_folds).parameters.keys())
    assert parameters == ALLOWED_PARAMETERS, (
        "assign_folds signature changed. It may only ever receive record IDs, "
        f"record-level labels and configuration. Got: {sorted(parameters)}"
    )


def test_assign_folds_has_no_segment_shaped_parameter_names():
    module = load_module()
    for name in inspect.signature(module.assign_folds).parameters.keys():
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in name.lower(), (
                f"assign_folds parameter '{name}' looks segment-derived"
            )


def test_split_helpers_also_take_only_records_and_labels():
    """The helpers assign_folds delegates to must not widen the contract either."""
    module = load_module()
    for helper in [module.assign_dev_partition, module.assign_cv_folds]:
        for name in inspect.signature(helper).parameters.keys():
            for forbidden in FORBIDDEN_SUBSTRINGS:
                assert forbidden not in name.lower(), (
                    f"{helper.__name__} parameter '{name}' looks segment-derived"
                )


def test_assign_folds_is_deterministic_for_a_fixed_seed():
    module = load_module()
    record_ids = ["a" + str(i).zfill(4) for i in range(100)]
    labels = [1 if i % 3 else 0 for i in range(100)]

    first = module.assign_folds(record_ids, labels, 42, 5, 0.6, 0.15, 0.25, 0.15)
    second = module.assign_folds(record_ids, labels, 42, 5, 0.6, 0.15, 0.25, 0.15)
    pd.testing.assert_frame_equal(first, second)


def test_assign_folds_output_is_record_level_and_disjoint():
    module = load_module()
    record_ids = ["a" + str(i).zfill(4) for i in range(120)]
    labels = [1 if i % 3 else 0 for i in range(120)]
    table = module.assign_folds(record_ids, labels, 7, 5, 0.6, 0.15, 0.25, 0.15)

    assert table["record_id"].nunique() == len(table)
    assert set(table["dev_partition"].unique()) == {"train", "val", "test"}
    for fold_index in range(5):
        column = "cv_inner_val_fold_" + str(fold_index)
        clash = table[(table["cv_fold"] == fold_index) & table[column]]
        assert len(clash) == 0


def test_assign_folds_ignores_record_id_ordering():
    """The split must depend on the seed, not on the order rows happen to arrive.

    A split that changes with input ordering is not reproducible, and filesystem
    iteration order is exactly what broke the old pipeline's augmentation.
    """
    module = load_module()
    record_ids = ["a" + str(i).zfill(4) for i in range(90)]
    labels = [1 if i % 3 else 0 for i in range(90)]

    forward = module.assign_folds(record_ids, labels, 42, 5, 0.6, 0.15, 0.25, 0.15)
    pairs = list(zip(record_ids, labels))[::-1]
    reversed_ids = [pair[0] for pair in pairs]
    reversed_labels = [pair[1] for pair in pairs]
    backward = module.assign_folds(reversed_ids, reversed_labels, 42, 5, 0.6, 0.15, 0.25, 0.15)

    merged = forward.merge(backward, on="record_id", suffixes=("_fwd", "_bwd"))
    assert len(merged) == len(forward)
    assert (merged["dev_partition_fwd"] == merged["dev_partition_bwd"]).all(), (
        "the development partition changed when the input order was reversed"
    )
    assert (merged["cv_fold_fwd"] == merged["cv_fold_bwd"]).all(), (
        "CV fold assignment changed when the input order was reversed"
    )


def test_the_split_script_never_reads_a_segment_index():
    """Belt and braces: the script itself must not open a segment-level file."""
    source = open(SCRIPT_PATH, encoding="utf-8").read()
    for forbidden in ["segments_index", "augmented_index", "row_index", "scalogram"]:
        assert forbidden not in source, (
            f"05_assign_folds.py references '{forbidden}'. The split may only see "
            "record IDs and record-level labels."
        )
