"""Convert PhysioNet 2016 Training-A WFDB records to one .npz per record.

Reads the raw WFDB pairs, keeps only the records that carry both an ECG and a
PCG channel, normalizes PCG per record by its own max absolute value, and
writes ecg / pcg / fs into data/interim/records/.

The 4 PCG-only records have a .hea but no .dat and are excluded.
The .wav files duplicate the PCG channel and are ignored - wfdb reads both
channels straight from the header.
"""

import argparse
import os

import numpy as np
import pandas as pd
import wfdb
import yaml

ECG_CHANNEL_NAME = "ECG"
PCG_CHANNEL_NAME = "PCG"
REFERENCE_FILE = "REFERENCE.csv"
RECORDS_INDEX_NAME = "records_index.csv"
EXCLUDED_LOG_NAME = "excluded_records.csv"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def load_labels(raw_dir):
    """Read REFERENCE.csv and map the challenge labels {-1, +1} to {0, 1}."""
    reference_path = os.path.join(raw_dir, REFERENCE_FILE)
    frame = pd.read_csv(reference_path, header=None, names=["record_id", "raw_label"])
    frame["label"] = (frame["raw_label"] > 0).astype(int)
    return frame


def find_channel_index(signal_names, wanted):
    """Return the index of a named channel, or -1 if the record does not carry it."""
    matches = [i for i, name in enumerate(signal_names) if name.strip().upper() == wanted]
    if len(matches) == 0:
        return -1
    return matches[0]


def read_record(raw_dir, record_id):
    """Read one WFDB record and return (ecg, pcg, fs), or None if not dual-modality."""
    record = wfdb.rdrecord(os.path.join(raw_dir, record_id))
    ecg_index = find_channel_index(record.sig_name, ECG_CHANNEL_NAME)
    pcg_index = find_channel_index(record.sig_name, PCG_CHANNEL_NAME)
    if ecg_index < 0 or pcg_index < 0:
        return None
    ecg = record.p_signal[:, ecg_index].astype(np.float32)
    pcg = record.p_signal[:, pcg_index].astype(np.float32)
    return ecg, pcg, int(record.fs)


def normalize_pcg(pcg):
    """Scale PCG to [-1, 1] by its own max absolute value.

    Per-record by construction, so no cross-record statistic exists and there is
    nothing here that could leak between folds.
    """
    peak = np.nanmax(np.abs(pcg))
    if peak <= 0 or not np.isfinite(peak):
        return pcg
    return (pcg / peak).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="WFDB -> per-record .npz")
    parser.add_argument("--raw-dir", required=True, help="raw training-a folder")
    parser.add_argument("--output-dir", required=True, help="where per-record .npz go")
    parser.add_argument("--index-out", required=True, help="records_index.csv path")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    seed_note = params["global_seed"]
    convert_params = params["data"]["convert"]
    expected_fs = convert_params["expected_fs"]
    declared_pcg_only = set(convert_params["pcg_only_records"])
    smoke = params["smoke"]

    print("=== 01_convert ===")
    print(f"raw_dir={args.raw_dir} expected_fs={expected_fs} global_seed={seed_note}")

    labels = load_labels(args.raw_dir)
    print(f"REFERENCE.csv rows: {len(labels)}")
    print(f"label balance in reference: normal={int((labels['label'] == 0).sum())} abnormal={int((labels['label'] == 1).sum())}")

    if smoke["enabled"]:
        labels = labels.head(smoke["n_records"]).copy()
        print(f"SMOKE MODE: limited to first {len(labels)} records")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    rows = []
    excluded = []
    for record_id, label in zip(labels["record_id"], labels["label"], strict=True):
        result = read_record(args.raw_dir, record_id)
        if result is None:
            excluded.append({"record_id": record_id, "reason": "no_ecg_channel"})
            continue
        ecg, pcg, fs = result
        if fs != expected_fs:
            excluded.append({"record_id": record_id, "reason": f"unexpected_fs_{fs}"})
            continue
        pcg = normalize_pcg(pcg)
        np.savez_compressed(
            os.path.join(args.output_dir, record_id + ".npz"),
            ecg=ecg,
            pcg=pcg,
            fs=np.int32(fs),
        )
        rows.append(
            {"record_id": record_id, "label": int(label), "n_samples": int(len(ecg)), "fs": fs}
        )

    index = pd.DataFrame(rows)
    index.to_csv(args.index_out, index=False)

    excluded_frame = pd.DataFrame(excluded)
    excluded_path = os.path.join(os.path.dirname(args.index_out), EXCLUDED_LOG_NAME)
    excluded_frame.to_csv(excluded_path, index=False)

    excluded_ids = set(excluded_frame["record_id"]) if len(excluded_frame) else set()
    print(f"converted {len(index)} records, excluded {len(excluded_frame)}")
    if len(excluded_frame):
        print(f"excluded ids: {sorted(excluded_ids)}")
    print(f"kept label balance: normal={int((index['label'] == 0).sum())} abnormal={int((index['label'] == 1).sum())}")

    if len(index) == 0:
        raise SystemExit("QC FAIL: 01_convert produced zero records")
    if not smoke["enabled"] and excluded_ids != declared_pcg_only:
        raise SystemExit(
            "QC FAIL: excluded set "
            + str(sorted(excluded_ids))
            + " does not match the declared PCG-only records "
            + str(sorted(declared_pcg_only))
        )

    print(f"wrote {args.index_out} and {excluded_path}")


if __name__ == "__main__":
    main()
