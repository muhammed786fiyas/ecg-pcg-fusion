"""Record-level train/val split of the pretraining records - early stopping only.

split_records() sees (record_id, stratum) pairs and a seed and nothing else,
the same contract as scripts/data/05_assign_folds.py: no window ID, no signal,
nothing derived from one. The stratum is subset + label, so each hospital's
share of normal and abnormal recordings is kept in both partitions. Records are
sorted first so the split depends on the record SET and the seed only.

There is no test partition: nothing about pretraining is a paper result. The
paper's numbers come from fine-tuning on Training-A's record-level CV folds,
which this pipeline never reads.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

MANIFEST_COLUMNS = ["row_index", "segment_variant_id", "record_id", "label"]


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def split_records(record_ids, strata, seed, val_frac):
    """The set of validation record IDs, from record IDs and strata alone."""
    pairs = sorted(zip([str(value) for value in record_ids], [str(value) for value in strata], strict=True))
    ids = np.asarray([pair[0] for pair in pairs])
    groups = np.asarray([pair[1] for pair in pairs])
    val_ids = train_test_split(ids, test_size=val_frac, random_state=seed, stratify=groups, shuffle=True)[1]
    return set(val_ids.tolist())


def main():
    parser = argparse.ArgumentParser(description="record-level split of the pretraining records")
    parser.add_argument("--records-index", required=True, help="pretrain 01_convert index")
    parser.add_argument("--row-index", required=True, help="pretrain 02_scalogram row_index.csv")
    parser.add_argument("--output-dir", required=True, help="data/processed/manifests_pcg_pretrain")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    seed = params["global_seed"]
    val_frac = params["pretrain"]["val_frac"]

    print("=== pretrain 03_split ===")
    records = pd.read_csv(args.records_index)
    rows = pd.read_csv(args.row_index)
    # Split over the records that actually have windows, as Training-A splits
    # over the records that survive segmentation.
    records = records[records["record_id"].isin(set(rows["record_id"]))]
    strata = (records["subset"] + "_" + records["label"].astype(str)).tolist()
    counts = pd.Series(strata).value_counts()
    if int(counts.min()) < 2:
        raise SystemExit(f"QC FAIL: a stratum has fewer than 2 records: {counts.to_dict()}")
    print(f"records={len(records)} val_frac={val_frac} seed={seed} strata={counts.to_dict()}")

    val_ids = split_records(records["record_id"].tolist(), strata, seed, val_frac)
    table = records[["record_id", "subset", "label"]].copy()
    table["partition"] = ["val" if record_id in val_ids else "train" for record_id in table["record_id"]]

    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, "pretrain_split.csv"), index=False)

    partition_of = dict(zip(table["record_id"], table["partition"], strict=True))
    rows["partition"] = [partition_of[record_id] for record_id in rows["record_id"]]
    for name in ["train", "val"]:
        subset = rows[rows["partition"] == name]
        if subset["label"].nunique() < 2:
            raise SystemExit(f"QC FAIL: pretraining {name} partition has only one class")
        overlap = set(subset["record_id"]) & set(rows[rows["partition"] != name]["record_id"])
        if len(overlap) > 0:
            raise SystemExit(f"QC FAIL: {len(overlap)} records appear in both partitions")
        subset[MANIFEST_COLUMNS].to_csv(os.path.join(args.output_dir, name + ".csv"), index=False)
        records_in = table[table["partition"] == name]
        print(f"  {name}: {len(records_in)} records, {len(subset)} windows, "
              f"abnormal share {round(float(subset['label'].mean()), 3)}")
    print(f"wrote train.csv, val.csv and pretrain_split.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
