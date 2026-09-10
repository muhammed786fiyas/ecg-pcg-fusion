"""THE SPLIT. Record-level, stratified, seeded.

The governing rule: the patient is the unit of splitting, and the split function
may only ever see record IDs and record-level labels. assign_folds() below takes
exactly (record_ids, labels, seed, ...) and nothing else - no segment ID, no
signal, no quantity derived from one. That is structural, not a convention, and
tests/test_split_inputs.py asserts it by inspecting the signature.

Output is one auditable table, fold_assignments.csv:
    record_id, label, dev_partition, cv_fold, cv_inner_val_fold_0 .. _<k-1>

  dev_partition  in {train, val, test}   - the development protocol (fold 0 only)
  cv_fold        in {0 .. k-1}           - which CV fold holds this record OUT as test
  cv_inner_val_fold_<i>  bool            - for fold i, is this record inner-validation?

For fold i the training pool is every record with cv_fold != i; roughly
cv_inner_val_frac of that pool is held out as inner-validation for early
stopping. Test records for fold i are never inner-validation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

DEV_TRAIN = "train"
DEV_VAL = "val"
DEV_TEST = "test"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def assign_dev_partition(record_ids, labels, seed, train_frac, val_frac, test_frac):
    """Stratified record-level train/val/test partition for the development protocol.

    Sees only record IDs, labels and a seed.
    """
    ids = np.asarray(record_ids)
    targets = np.asarray(labels)

    train_ids, holdout_ids, train_labels, holdout_labels = train_test_split(
        ids,
        targets,
        test_size=(val_frac + test_frac),
        random_state=seed,
        stratify=targets,
        shuffle=True,
    )
    val_share = val_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=holdout_labels,
        shuffle=True,
    )

    partition = {}
    for record_id in train_ids:
        partition[record_id] = DEV_TRAIN
    for record_id in val_ids:
        partition[record_id] = DEV_VAL
    for record_id in test_ids:
        partition[record_id] = DEV_TEST
    return partition


def assign_cv_folds(record_ids, labels, seed, k):
    """Stratified record-level k-fold. Returns record_id -> held-out test fold index."""
    ids = np.asarray(record_ids)
    targets = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_of = {}
    for fold_index, split in enumerate(splitter.split(ids, targets)):
        test_positions = split[1]
        for position in test_positions:
            fold_of[ids[position]] = fold_index
    return fold_of


def assign_inner_val(record_ids, labels, fold_of, fold_index, seed, inner_val_frac):
    """Records held out of fold_index's training pool for early stopping.

    Stratified over the training pool only. Test records of this fold are never
    inner-validation.
    """
    pool_ids = [rid for rid in record_ids if fold_of[rid] != fold_index]
    label_of = dict(zip(record_ids, labels))
    pool_labels = [label_of[rid] for rid in pool_ids]

    inner_ids = train_test_split(
        np.asarray(pool_ids),
        test_size=inner_val_frac,
        random_state=(seed + fold_index),
        stratify=np.asarray(pool_labels),
        shuffle=True,
    )[1]
    return set(inner_ids)


def assign_folds(record_ids, labels, seed, k, dev_train_frac, dev_val_frac, dev_test_frac, cv_inner_val_frac):
    """The whole split, from record IDs and labels alone.

    This signature is the contract: nothing segment-derived can reach it.
    """
    ids = list(record_ids)
    targets = [int(value) for value in labels]

    partition = assign_dev_partition(ids, targets, seed, dev_train_frac, dev_val_frac, dev_test_frac)
    fold_of = assign_cv_folds(ids, targets, seed, k)

    table = pd.DataFrame(
        {
            "record_id": ids,
            "label": targets,
            "dev_partition": [partition[rid] for rid in ids],
            "cv_fold": [fold_of[rid] for rid in ids],
        }
    )

    for fold_index in range(k):
        inner_ids = assign_inner_val(ids, targets, fold_of, fold_index, seed, cv_inner_val_frac)
        table["cv_inner_val_fold_" + str(fold_index)] = [rid in inner_ids for rid in ids]

    return table


def report_balance(table, k):
    print("development protocol:")
    for name in [DEV_TRAIN, DEV_VAL, DEV_TEST]:
        subset = table[table["dev_partition"] == name]
        abnormal = int((subset["label"] == 1).sum())
        normal = int((subset["label"] == 0).sum())
        share = round(100.0 * len(subset) / len(table), 1)
        print(f"  {name}: {len(subset)} records ({share}%) normal={normal} abnormal={abnormal}")

    print(f"cross-validation protocol ({k} folds):")
    for fold_index in range(k):
        test = table[table["cv_fold"] == fold_index]
        inner = table[table["cv_inner_val_fold_" + str(fold_index)]]
        train = table[(table["cv_fold"] != fold_index) & (~table["cv_inner_val_fold_" + str(fold_index)])]
        print(
            f"  fold {fold_index}: train={len(train)} inner_val={len(inner)} test={len(test)}"
            f"  test abnormal share={round(float(test['label'].mean()), 3)}"
        )


def check_no_overlap(table, k):
    """Hard-fail if any record lands in two partitions of the same protocol."""
    counts = table["record_id"].value_counts()
    if int(counts.max()) > 1:
        raise SystemExit("QC FAIL: duplicate record_id in fold_assignments")

    for fold_index in range(k):
        column = "cv_inner_val_fold_" + str(fold_index)
        clash = table[(table["cv_fold"] == fold_index) & table[column]]
        if len(clash) > 0:
            raise SystemExit(
                f"QC FAIL: {len(clash)} records are both test and inner-validation in fold {fold_index}"
            )


def main():
    parser = argparse.ArgumentParser(description="record-level stratified fold assignment")
    parser.add_argument("--index-in", required=True, help="post-QC record index")
    parser.add_argument("--output", required=True, help="fold_assignments.csv path")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    seed = params["global_seed"]
    split_params = params["split"]
    k = params["evaluation"]["cv_folds"]
    if params["smoke"]["enabled"]:
        k = params["smoke"]["cv_folds"]

    print("=== 05_assign_folds ===")
    print(f"global_seed={seed} cv_folds={k} inner_val_frac={split_params['cv_inner_val_frac']}")

    index = pd.read_csv(args.index_in)

    # The only two columns that ever reach the split function.
    record_ids = index["record_id"].tolist()
    labels = index["label"].tolist()
    print(f"records to split: {len(record_ids)} normal={labels.count(0)} abnormal={labels.count(1)}")

    if len(record_ids) != len(set(record_ids)):
        raise SystemExit("QC FAIL: duplicate record_id in the input index")
    if k < 2:
        raise SystemExit(f"QC FAIL: cv_folds must be at least 2, got {k}")
    smallest_class = min(labels.count(0), labels.count(1))
    if smallest_class < k:
        raise SystemExit(f"QC FAIL: smallest class has {smallest_class} records, fewer than {k} folds")

    table = assign_folds(
        record_ids,
        labels,
        seed,
        k,
        split_params["dev_train_frac"],
        split_params["dev_val_frac"],
        split_params["dev_test_frac"],
        split_params["cv_inner_val_frac"],
    )

    check_no_overlap(table, k)
    report_balance(table, k)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {args.output} with {len(table)} records and {k} folds")


if __name__ == "__main__":
    main()
