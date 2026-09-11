"""Convert PhysioNet 2016 subsets B-F (PCG only) to one .npz per record.

PCG-branch pretraining data, added 2026-09-12. Subsets B-F were recorded at
different hospitals, on different patients, from Training-A, so no recording
here is a Training-A patient and nothing in this pipeline can reach a CV test
record. Training-A is never read; reject_training_a() below hard-fails if a
Training-A record ID ever turns up.

Each PCG is processed exactly as Training-A's is in scripts/data/01_convert.py
and scripts/data/02_record_qc.py: normalised by its own max absolute value,
then gated on NaN fraction, flatness and length with the same thresholds from
params.yaml. The helpers are duplicated rather than imported, by convention.
"""

import argparse
import os

import numpy as np
import pandas as pd
import wfdb
import yaml

PCG_CHANNEL_NAME = "PCG"
REFERENCE_FILE = "REFERENCE.csv"
DROPPED_LOG_NAME = "pretrain_records_dropped.csv"
TRAINING_A_PREFIX = "a"


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def load_labels(subset_dir):
    """Read REFERENCE.csv and map the challenge labels {-1, +1} to {0, 1}."""
    frame = pd.read_csv(os.path.join(subset_dir, REFERENCE_FILE), header=None, names=["record_id", "raw_label"])
    frame["label"] = (frame["raw_label"] > 0).astype(int)
    return frame


def reject_training_a(record_ids):
    """Structural guard: Training-A records are the CV data and must never be
    pretrained on. Their IDs start with 'a'; B-F IDs start with b-f."""
    leaked = [str(record_id) for record_id in record_ids if str(record_id).lower().startswith(TRAINING_A_PREFIX)]
    if len(leaked) > 0:
        raise SystemExit(f"QC FAIL: Training-A records reached the pretraining pipeline: {leaked[:5]}")


def find_channel_index(signal_names, wanted):
    matches = [i for i, name in enumerate(signal_names) if name.strip().upper() == wanted]
    if len(matches) == 0:
        return -1
    return matches[0]


def read_pcg(subset_dir, record_id):
    """Return (pcg, fs), or None if the record carries no PCG channel."""
    record = wfdb.rdrecord(os.path.join(subset_dir, record_id))
    pcg_index = find_channel_index(record.sig_name, PCG_CHANNEL_NAME)
    if pcg_index < 0:
        return None
    return record.p_signal[:, pcg_index].astype(np.float32), int(record.fs)


def normalize_pcg(pcg):
    """Scale PCG to [-1, 1] by its own max absolute value - as Training-A."""
    peak = np.nanmax(np.abs(pcg))
    if peak <= 0 or not np.isfinite(peak):
        return pcg
    return (pcg / peak).astype(np.float32)


def nan_fraction(signal):
    if len(signal) == 0:
        return 1.0
    return float(np.mean(~np.isfinite(signal)))


def finite_std(signal):
    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return 0.0
    return float(np.std(finite))


def check_pcg(signal, fs, max_nan_fraction, min_std, min_duration_s):
    """Training-A's record QC for one channel: a reason string, or '' if it passes."""
    if len(signal) < int(min_duration_s * fs):
        return "pcg_too_short"
    fraction = nan_fraction(signal)
    if fraction > max_nan_fraction:
        return f"pcg_majority_nan_{round(fraction, 3)}"
    if finite_std(signal) < min_std:
        return "pcg_flat"
    return ""


def main():
    parser = argparse.ArgumentParser(description="PhysioNet 2016 B-F PCG -> per-record .npz")
    parser.add_argument("--raw-dir", required=True, help="folder holding training-b ... training-f")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    subsets = params["pretrain"]["subsets"]
    expected_fs = params["data"]["convert"]["expected_fs"]
    qc = params["data"]["record_qc"]

    print("=== pretrain 01_convert ===")
    print(f"subsets={subsets} expected_fs={expected_fs} record QC as Training-A: {qc}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    rows = []
    dropped = []
    for subset in subsets:
        subset_dir = os.path.join(args.raw_dir, subset)
        if not os.path.isdir(subset_dir):
            raise SystemExit(f"QC FAIL: subset folder not found: {subset_dir}")
        labels = load_labels(subset_dir)
        reject_training_a(labels["record_id"])

        for record_id, label in zip(labels["record_id"], labels["label"], strict=True):
            result = read_pcg(subset_dir, record_id)
            if result is None:
                dropped.append({"record_id": record_id, "subset": subset, "reason": "no_pcg_channel"})
                continue
            pcg, fs = result
            if fs != expected_fs:
                dropped.append({"record_id": record_id, "subset": subset, "reason": f"unexpected_fs_{fs}"})
                continue
            pcg = normalize_pcg(pcg)
            reason = check_pcg(pcg, fs, qc["max_nan_fraction"], qc["min_std"], qc["min_duration_s"])
            if reason != "":
                dropped.append({"record_id": record_id, "subset": subset, "reason": reason})
                continue
            np.savez_compressed(os.path.join(args.output_dir, record_id + ".npz"), pcg=pcg, fs=np.int32(fs))
            rows.append({"record_id": record_id, "subset": subset, "label": int(label),
                         "n_samples": int(len(pcg)), "fs": fs})
        kept = [row for row in rows if row["subset"] == subset]
        print(f"  {subset}: {len(labels)} in REFERENCE.csv, kept {len(kept)}, "
              f"abnormal {sum([row['label'] for row in kept])}")

    index = pd.DataFrame(rows)
    dropped_frame = pd.DataFrame(dropped)
    if len(index) == 0:
        raise SystemExit("QC FAIL: no pretraining records survived")
    if index["record_id"].nunique() != len(index):
        raise SystemExit("QC FAIL: duplicate record IDs across subsets")
    if index["label"].nunique() < 2:
        raise SystemExit("QC FAIL: pretraining data has only one class")

    index.to_csv(args.index_out, index=False)
    dropped_path = os.path.join(os.path.dirname(args.index_out), DROPPED_LOG_NAME)
    dropped_frame.to_csv(dropped_path, index=False)

    hours = round(float(index["n_samples"].sum()) / expected_fs / 3600.0, 2)
    print(f"kept {len(index)} records ({hours} h of PCG), dropped {len(dropped_frame)}")
    if len(dropped_frame):
        for reason, count in dropped_frame["reason"].value_counts().items():
            print(f"  drop reason {reason}: {count}")
    print(f"label balance: normal={int((index['label'] == 0).sum())} abnormal={int((index['label'] == 1).sum())}")
    print(f"wrote {args.index_out} and {dropped_path}")


if __name__ == "__main__":
    main()
