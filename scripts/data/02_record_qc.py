"""Record-level quality gate, run before any splitting.

Drops records whose ECG or PCG is majority-NaN, flat/constant, or too short to
yield a single 3-second window.

This runs at record level, not only per segment as the old pipeline did, so a
record that would silently contribute zero segments is removed before fold
assignment rather than after. Otherwise it still counts toward fold sizes and
skews the class balance.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

QC_LOG_NAME = "record_qc_dropped.csv"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def nan_fraction(signal):
    if len(signal) == 0:
        return 1.0
    return float(np.mean(~np.isfinite(signal)))


def finite_std(signal):
    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return 0.0
    return float(np.std(finite))


def check_channel(signal, fs, name, max_nan_fraction, min_std, min_duration_s):
    """Return a failure reason string, or an empty string if the channel passes."""
    if len(signal) < int(min_duration_s * fs):
        return f"{name}_too_short"
    frac = nan_fraction(signal)
    if frac > max_nan_fraction:
        return f"{name}_majority_nan_{round(frac, 3)}"
    if finite_std(signal) < min_std:
        return f"{name}_flat"
    return ""


def main():
    parser = argparse.ArgumentParser(description="record-level QC gate")
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    qc = params["data"]["record_qc"]
    max_nan_fraction = qc["max_nan_fraction"]
    min_std = qc["min_std"]
    min_duration_s = qc["min_duration_s"]

    print("=== 02_record_qc ===")
    print(f"max_nan_fraction={max_nan_fraction} min_std={min_std} min_duration_s={min_duration_s}")

    index = pd.read_csv(args.index_in)
    print(f"records in: {len(index)}")

    kept = []
    dropped = []
    for row in index.itertuples(index=False):
        payload = np.load(os.path.join(args.records_dir, row.record_id + ".npz"))
        ecg = payload["ecg"]
        pcg = payload["pcg"]
        fs = int(payload["fs"])

        reason = check_channel(ecg, fs, "ecg", max_nan_fraction, min_std, min_duration_s)
        if reason == "":
            reason = check_channel(pcg, fs, "pcg", max_nan_fraction, min_std, min_duration_s)

        if reason == "":
            kept.append(row._asdict())
        else:
            dropped.append({"record_id": row.record_id, "label": int(row.label), "reason": reason})

    kept_frame = pd.DataFrame(kept)
    dropped_frame = pd.DataFrame(dropped)

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    kept_frame.to_csv(args.index_out, index=False)
    dropped_path = os.path.join(os.path.dirname(args.index_out), QC_LOG_NAME)
    dropped_frame.to_csv(dropped_path, index=False)

    print(f"records kept: {len(kept_frame)}  dropped: {len(dropped_frame)}")
    if len(dropped_frame):
        for reason, count in dropped_frame["reason"].value_counts().items():
            print(f"  drop reason {reason}: {count}")
        print(f"  dropped ids: {sorted(dropped_frame['record_id'])}")
    print(f"kept label balance: normal={int((kept_frame['label'] == 0).sum())} abnormal={int((kept_frame['label'] == 1).sum())}")

    if len(kept_frame) == 0:
        raise SystemExit("QC FAIL: record QC dropped every record")
    if kept_frame["label"].nunique() < 2:
        raise SystemExit("QC FAIL: record QC left only one class")

    print(f"wrote {args.index_out} and {dropped_path}")


if __name__ == "__main__":
    main()
