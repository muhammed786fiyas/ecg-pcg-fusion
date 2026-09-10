"""The leakage guards. This is the point of the suite.

Split bugs come back during refactors, and they are invisible: the code runs
fine and the number is just quietly optimistic. These assert the invariants
against the manifests that were actually emitted, not against the intent.
"""

import os

import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_ROOT = os.path.join(REPO_ROOT, "data", "processed", "manifests")
PARTITIONS = ["train", "val", "test"]
ORIG_SUFFIX = "_orig"


def load_cv_folds():
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        params = yaml.safe_load(handle)
    return params["evaluation"]["cv_folds"]


def protocol_tags():
    tags = ["dev"]
    for fold_index in range(load_cv_folds()):
        tags.append("cv_fold" + str(fold_index))
    return tags


def manifest_dir(tag):
    return os.path.join(MANIFEST_ROOT, tag)


def read_manifests(tag):
    directory = manifest_dir(tag)
    if not os.path.isdir(directory):
        pytest.skip(f"manifests for {tag} not built yet")
    return {name: pd.read_csv(os.path.join(directory, name + ".csv")) for name in PARTITIONS}


@pytest.mark.parametrize("tag", protocol_tags())
def test_no_record_in_two_partitions(tag):
    """A record must never appear in more than one of train/val/test.

    This is the top correctness requirement of the whole project.
    """
    manifests = read_manifests(tag)
    record_sets = {name: set(frame["record_id"]) for name, frame in manifests.items()}

    for left in range(len(PARTITIONS)):
        for right in range(left + 1, len(PARTITIONS)):
            name_a = PARTITIONS[left]
            name_b = PARTITIONS[right]
            overlap = record_sets[name_a] & record_sets[name_b]
            assert len(overlap) == 0, (
                f"{tag}: {len(overlap)} record(s) in both {name_a} and {name_b}: "
                + str(sorted(overlap)[:5])
            )


@pytest.mark.parametrize("tag", protocol_tags())
def test_no_base_segment_in_two_partitions(tag):
    """Nor may a base segment, via any of its augmented variants."""
    manifests = read_manifests(tag)
    seen = {}
    for name in PARTITIONS:
        for variant_id in manifests[name]["segment_variant_id"]:
            base = variant_id.rsplit("_", 1)[0]
            if base in seen:
                assert seen[base] == name, (
                    f"{tag}: segment {base} appears in both {seen[base]} and {name}"
                )
            seen[base] = name


@pytest.mark.parametrize("tag", protocol_tags())
def test_val_and_test_are_orig_only(tag):
    """Validation and test are never augmented.

    Validation exists to estimate real-data performance so early stopping picks
    the right checkpoint. Augmented copies of the same segment are near
    duplicates, and they make the metric look more precise than it is.
    """
    manifests = read_manifests(tag)
    for name in ["val", "test"]:
        offenders = [
            value for value in manifests[name]["segment_variant_id"] if not value.endswith(ORIG_SUFFIX)
        ]
        assert len(offenders) == 0, (
            f"{tag}: {name} manifest contains {len(offenders)} augmented rows, e.g. {offenders[:3]}"
        )


@pytest.mark.parametrize("tag", protocol_tags())
def test_train_carries_all_variants(tag):
    """The training manifest should carry the full 4x expansion.

    The mirror of the previous test: if this ever collapses to _orig only, the
    augmentation silently stopped being used and nothing else would notice.
    """
    manifests = read_manifests(tag)
    suffixes = set([value.rsplit("_", 1)[1] for value in manifests["train"]["segment_variant_id"]])
    assert suffixes == {"orig", "noise", "scale", "combined"}, (
        f"{tag}: train manifest variants are {sorted(suffixes)}"
    )


@pytest.mark.parametrize("tag", protocol_tags())
def test_manifests_are_non_empty_and_two_class(tag):
    manifests = read_manifests(tag)
    for name in PARTITIONS:
        frame = manifests[name]
        assert len(frame) > 0, f"{tag}: {name} manifest is empty"
        assert set(frame["label"].unique()) == {0, 1}, (
            f"{tag}: {name} manifest is not two-class, labels are {sorted(frame['label'].unique())}"
        )


@pytest.mark.parametrize("tag", protocol_tags())
def test_row_indices_are_unique_within_partition(tag):
    """A duplicated memmap row would silently double-count a segment."""
    manifests = read_manifests(tag)
    for name in PARTITIONS:
        rows = manifests[name]["row_index"]
        assert rows.nunique() == len(rows), f"{tag}: {name} has duplicate row_index values"


def test_fold_assignments_cover_every_record_exactly_once():
    path = os.path.join(MANIFEST_ROOT, "fold_assignments.csv")
    if not os.path.exists(path):
        pytest.skip("fold_assignments.csv not built yet")
    table = pd.read_csv(path)
    assert table["record_id"].nunique() == len(table), "duplicate record_id in fold_assignments"
    assert set(table["dev_partition"].unique()) == set(PARTITIONS)
    assert sorted(table["cv_fold"].unique()) == list(range(load_cv_folds()))


def test_cv_test_records_are_never_inner_validation():
    """For fold i, a record held out as test cannot also be early-stopping data."""
    path = os.path.join(MANIFEST_ROOT, "fold_assignments.csv")
    if not os.path.exists(path):
        pytest.skip("fold_assignments.csv not built yet")
    table = pd.read_csv(path)
    for fold_index in range(load_cv_folds()):
        column = "cv_inner_val_fold_" + str(fold_index)
        clash = table[(table["cv_fold"] == fold_index) & table[column]]
        assert len(clash) == 0, f"fold {fold_index}: {len(clash)} records are both test and inner-val"


def test_every_record_is_tested_exactly_once_across_cv_folds():
    """The defining property of k-fold CV, and easy to break with an off-by-one."""
    path = os.path.join(MANIFEST_ROOT, "fold_assignments.csv")
    if not os.path.exists(path):
        pytest.skip("fold_assignments.csv not built yet")
    table = pd.read_csv(path)
    counts = table["cv_fold"].value_counts()
    assert int(counts.sum()) == len(table)
    assert len(counts) == load_cv_folds()
