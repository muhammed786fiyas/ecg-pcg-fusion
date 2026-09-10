"""Build the training manifests. These are the ONLY thing a training script reads.

Joins fold_assignments.csv (record -> partition) with the scalogram row index
(segment_variant_id -> memmap row) and emits, per protocol and per fold, three
CSVs listing row_index, segment_variant_id, record_id, label.

Two rules are enforced here rather than trusted:

1. Train manifests carry all four augmentation variants. Val and test manifests
   carry _orig rows ONLY. Validation exists to estimate real-data performance so
   early stopping picks the right checkpoint, and augmented copies of the same
   segment are near-duplicates that make the metric look more precise than it
   is. This is the level at which the train-only augmentation rule lives.

2. No record_id may appear in more than one partition of the same protocol. The
   split itself is record-level by construction, but a join can still go wrong,
   so the invariant is re-checked against the emitted manifests and hard-fails.

Negative-control protocols (leaky_val, leaky_all) are built by
scripts/modeling/split_protocol/, not here - this script only ever produces
correct record-level manifests.
"""

import argparse
import os

import pandas as pd
import yaml

TRAIN = "train"
VAL = "val"
TEST = "test"
PARTITIONS = [TRAIN, VAL, TEST]
ORIG_VARIANT = "orig"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def select_rows(rows, record_ids, orig_only):
    """Rows belonging to the given records, optionally restricted to _orig."""
    subset = rows[rows["record_id"].isin(record_ids)]
    if orig_only:
        subset = subset[subset["variant"] == ORIG_VARIANT]
    return subset[["row_index", "segment_variant_id", "record_id", "label"]].copy()


def dev_partitions(folds):
    train_ids = set(folds[folds["dev_partition"] == TRAIN]["record_id"])
    val_ids = set(folds[folds["dev_partition"] == VAL]["record_id"])
    test_ids = set(folds[folds["dev_partition"] == TEST]["record_id"])
    return {TRAIN: train_ids, VAL: val_ids, TEST: test_ids}


def cv_partitions(folds, fold_index):
    """For fold i: test = cv_fold == i, inner-val = the flagged records, train = the rest."""
    inner_column = "cv_inner_val_fold_" + str(fold_index)
    test_ids = set(folds[folds["cv_fold"] == fold_index]["record_id"])
    val_ids = set(folds[folds[inner_column]]["record_id"])
    train_ids = set(folds[(folds["cv_fold"] != fold_index) & (~folds[inner_column])]["record_id"])
    return {TRAIN: train_ids, VAL: val_ids, TEST: test_ids}


def check_disjoint(partition_ids, tag):
    for left in range(len(PARTITIONS)):
        for right in range(left + 1, len(PARTITIONS)):
            name_a = PARTITIONS[left]
            name_b = PARTITIONS[right]
            overlap = partition_ids[name_a] & partition_ids[name_b]
            if len(overlap) > 0:
                raise SystemExit(
                    f"QC FAIL: {tag} has {len(overlap)} record(s) in both {name_a} and {name_b}: "
                    + str(sorted(overlap)[:5])
                )


def write_manifests(rows, partition_ids, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name in PARTITIONS:
        orig_only = name != TRAIN
        manifest = select_rows(rows, partition_ids[name], orig_only)
        path = os.path.join(out_dir, name + ".csv")
        manifest.to_csv(path, index=False)
        written[name] = manifest

        n_records = manifest["record_id"].nunique()
        n_abnormal = int((manifest["label"] == 1).sum())
        share = round(float(manifest["label"].mean()), 3) if len(manifest) else 0.0
        print(
            f"  {tag} {name}: {len(manifest)} rows, {n_records} records, "
            f"abnormal rows={n_abnormal} (share {share}), orig_only={orig_only}"
        )
        if len(manifest) == 0:
            raise SystemExit(f"QC FAIL: {tag} {name} manifest is empty")

    # Re-check the augmentation rule against what was actually emitted.
    for name in [VAL, TEST]:
        ids = written[name]["segment_variant_id"]
        non_orig = [value for value in ids if not value.endswith("_" + ORIG_VARIANT)]
        if len(non_orig) > 0:
            raise SystemExit(f"QC FAIL: {tag} {name} manifest contains {len(non_orig)} augmented rows")

    # And that no base segment straddles two partitions.
    base_of = {}
    for name in PARTITIONS:
        for variant_id in written[name]["segment_variant_id"]:
            base = variant_id.rsplit("_", 1)[0]
            if base in base_of and base_of[base] != name:
                raise SystemExit(
                    f"QC FAIL: {tag} segment {base} appears in both {base_of[base]} and {name}"
                )
            base_of[base] = name

    return written


def main():
    parser = argparse.ArgumentParser(description="build train/val/test manifests")
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--row-index", required=True, help="scalogram row_index.csv")
    parser.add_argument("--output-dir", required=True, help="data/processed/manifests")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    k = params["evaluation"]["cv_folds"]
    if params["smoke"]["enabled"]:
        k = params["smoke"]["cv_folds"]

    print("=== 01_build_manifests ===")
    print(f"cv_folds={k}")

    folds = pd.read_csv(args.fold_assignments)
    rows = pd.read_csv(args.row_index)
    print(f"records: {len(folds)}  scalogram rows: {len(rows)}")

    missing = set(rows["record_id"]) - set(folds["record_id"])
    if len(missing) > 0:
        raise SystemExit(f"QC FAIL: {len(missing)} records in the row index have no fold assignment")

    print("development protocol:")
    dev_ids = dev_partitions(folds)
    check_disjoint(dev_ids, "dev")
    write_manifests(rows, dev_ids, os.path.join(args.output_dir, "dev"), "dev")

    for fold_index in range(k):
        print(f"cv fold {fold_index}:")
        fold_ids = cv_partitions(folds, fold_index)
        tag = "cv_fold" + str(fold_index)
        check_disjoint(fold_ids, tag)
        covered = len(fold_ids[TRAIN]) + len(fold_ids[VAL]) + len(fold_ids[TEST])
        if covered != len(folds):
            raise SystemExit(
                f"QC FAIL: {tag} covers {covered} records but there are {len(folds)}"
            )
        write_manifests(rows, fold_ids, os.path.join(args.output_dir, tag), tag)

    print(f"wrote manifests under {args.output_dir}")


if __name__ == "__main__":
    main()
