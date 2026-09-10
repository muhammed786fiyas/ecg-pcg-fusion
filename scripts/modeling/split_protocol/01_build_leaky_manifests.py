"""Build the deliberately leaky manifests for the split-protocol negative control.

*** THESE MANIFESTS ARE BROKEN ON PURPOSE. ***

They exist to quantify how much apparent performance is manufactured by split
choice alone. Everything they produce is a NEGATIVE CONTROL and must be labelled
as one in the CSV, in figure captions and in MLflow tags. Nothing here may ever
be read as a result of the method.

  Arm A - correct     : record-level train/val/test. Built by
                        dataset_prep/01_build_manifests.py, not here.
  Arm B - leaky val   : record-level test kept clean, but validation drawn by a
                        random split over SEGMENT ROWS from the training pool.
                        This is what the old pipeline did, and what a lot of
                        published work does implicitly. It corrupts early
                        stopping without touching the test set.
  Arm C - fully leaky : random segment-level split for validation AND test.
                        Segments of the same record appear in training and test.

Arms B and C still get all four augmentation variants in training and _orig-only
in val/test, so the ONLY thing that differs from arm A is the partitioning. That
is what makes the delta attributable to the split.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

TRAIN = "train"
VAL = "val"
TEST = "test"
ORIG_VARIANT = "orig"

BANNER = "*** NEGATIVE CONTROL - deliberately leaky split, not a result ***"


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def base_segment(variant_id):
    return variant_id.rsplit("_", 1)[0]


def split_rows_randomly(rows, fractions, seed):
    """Random split over SEGMENT rows. This is the leak, and it is intentional.

    Splitting on base segments rather than raw rows so a segment's four
    augmented variants stay together - otherwise the leak would be trivially
    total (a segment's own noise-augmented copy in training) rather than the
    realistic record-level leak we are trying to measure.
    """
    segments = np.array(sorted({base_segment(value) for value in rows["segment_variant_id"]}))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(segments)

    n_total = len(shuffled)
    cut_train = int(round(fractions[0] * n_total))
    cut_val = cut_train + int(round(fractions[1] * n_total))

    assignment = {}
    for segment in shuffled[:cut_train]:
        assignment[segment] = TRAIN
    for segment in shuffled[cut_train:cut_val]:
        assignment[segment] = VAL
    for segment in shuffled[cut_val:]:
        assignment[segment] = TEST

    rows = rows.copy()
    rows["partition"] = [assignment[base_segment(value)] for value in rows["segment_variant_id"]]
    return rows


def write_arm(rows_by_partition, out_dir, arm_name):
    os.makedirs(out_dir, exist_ok=True)
    for name in [TRAIN, VAL, TEST]:
        frame = rows_by_partition[name]
        if name != TRAIN:
            frame = frame[frame["variant"] == ORIG_VARIANT]
        frame = frame[["row_index", "segment_variant_id", "record_id", "label"]]
        frame.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
        print(
            f"  {arm_name} {name}: {len(frame)} rows, {frame['record_id'].nunique()} records, "
            f"abnormal share {round(float(frame['label'].mean()), 3)}"
        )

    with open(os.path.join(out_dir, "NEGATIVE_CONTROL.txt"), "w") as handle:
        handle.write(BANNER + "\n")
        handle.write(f"arm: {arm_name}\n")
        handle.write(
            "These manifests intentionally violate patient-level separation.\n"
            "They exist only to measure how much performance a leaky split manufactures.\n"
            "Never report a number from them as a result of the method.\n"
        )


def report_leak(rows_by_partition, arm_name):
    """Measure and print the leak, so the control's severity is on record."""
    train_records = set(rows_by_partition[TRAIN]["record_id"])
    test_records = set(rows_by_partition[TEST]["record_id"])
    val_records = set(rows_by_partition[VAL]["record_id"])

    train_test = len(train_records & test_records)
    train_val = len(train_records & val_records)
    print(
        f"  {arm_name} LEAK MEASURED: {train_val} record(s) shared between train and val, "
        f"{train_test} between train and test"
    )
    return {"arm": arm_name, "records_shared_train_val": train_val, "records_shared_train_test": train_test}


def main():
    parser = argparse.ArgumentParser(description="build the leaky negative-control manifests")
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--row-index", required=True)
    parser.add_argument("--output-root", default="data/processed/manifests_negative_control")
    parser.add_argument("--fold", type=int, default=0, help="which CV fold defines arm B's clean test set")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    seed = params["global_seed"]
    split_params = params["split"]

    print("=== 01_build_leaky_manifests ===")
    print(BANNER)
    print(f"fold={args.fold} seed={seed}")

    folds = pd.read_csv(args.fold_assignments)
    rows = pd.read_csv(args.row_index)

    fold_of = dict(zip(folds["record_id"], folds["cv_fold"], strict=True))
    rows = rows.copy()
    rows["cv_fold"] = [fold_of[value] for value in rows["record_id"]]

    leak_report = []

    # ---- Arm B: clean record-level test, validation leaked over segment rows ----
    print("arm B (leaky validation, clean test):")
    test_rows = rows[rows["cv_fold"] == args.fold]
    pool_rows = rows[rows["cv_fold"] != args.fold]

    inner_fraction = split_params["cv_inner_val_frac"]
    split_b = split_rows_randomly(pool_rows, [1.0 - inner_fraction, inner_fraction, 0.0], seed)
    arm_b = {
        TRAIN: split_b[split_b["partition"] == TRAIN],
        VAL: split_b[split_b["partition"] == VAL],
        TEST: test_rows,
    }
    write_arm(arm_b, os.path.join(args.output_root, "arm_b_leaky_val", "cv_fold" + str(args.fold)), "arm_b")
    leak_report.append(report_leak(arm_b, "arm_b_leaky_val"))

    # ---- Arm C: validation AND test both leaked over segment rows ----
    print("arm C (fully leaky):")
    fractions = [
        split_params["dev_train_frac"],
        split_params["dev_val_frac"],
        split_params["dev_test_frac"],
    ]
    split_c = split_rows_randomly(rows, fractions, seed)
    arm_c = {
        TRAIN: split_c[split_c["partition"] == TRAIN],
        VAL: split_c[split_c["partition"] == VAL],
        TEST: split_c[split_c["partition"] == TEST],
    }
    write_arm(arm_c, os.path.join(args.output_root, "arm_c_fully_leaky", "cv_fold" + str(args.fold)), "arm_c")
    leak_report.append(report_leak(arm_c, "arm_c_fully_leaky"))

    os.makedirs(args.output_root, exist_ok=True)
    pd.DataFrame(leak_report).to_csv(os.path.join(args.output_root, "leak_report.csv"), index=False)

    # A control that did not actually leak would silently make the whole
    # comparison meaningless, so fail loudly rather than produce a null result.
    if leak_report[0]["records_shared_train_val"] == 0:
        raise SystemExit("QC FAIL: arm B shares no records between train and val - it did not leak")
    if leak_report[1]["records_shared_train_test"] == 0:
        raise SystemExit("QC FAIL: arm C shares no records between train and test - it did not leak")

    print("")
    print(BANNER)
    print(f"wrote leaky manifests under {args.output_root}")


if __name__ == "__main__":
    main()
