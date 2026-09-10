"""Per-segment quality gate.

Drops a segment if either channel is more than max_nan_fraction non-finite, or
if any NaN/Inf survives linear interpolation. Segments that pass but contain a
few non-finite samples are repaired by interpolation, so nothing downstream has
to defend against NaN.

Every surviving segment is written to a NEW directory rather than edited in
place. Repairing 03's output in place would make this stage's output overlap its
own dependency, which DVC cannot track and which makes the stage impossible to
re-inspect: after one run the evidence of what was repaired is gone.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

DROPPED_LOG_NAME = "segment_qc_dropped.csv"
REPAIRED_LOG_NAME = "segment_qc_repaired.csv"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def nan_fraction(signal):
    if len(signal) == 0:
        return 1.0
    return float(np.mean(~np.isfinite(signal)))


def interpolate_non_finite(signal):
    """Linearly interpolate over non-finite samples. Returns a new array."""
    result = np.array(signal, dtype=np.float32, copy=True)
    bad = ~np.isfinite(result)
    if not bad.any():
        return result
    good = ~bad
    if not good.any():
        return result
    positions = np.arange(len(result))
    result[bad] = np.interp(positions[bad], positions[good], result[good]).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser(description="segment-level QC gate")
    parser.add_argument("--segments-dir", required=True, help="03_segment output, read-only")
    parser.add_argument("--output-dir", required=True, help="QC-passed segments go here")
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    max_nan_fraction = params["data"]["segment_qc"]["max_nan_fraction"]

    print("=== 04_segment_qc ===")
    print(f"max_nan_fraction={max_nan_fraction}")

    index = pd.read_csv(args.index_in)
    print(f"segments in: {len(index)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    kept = []
    dropped = []
    repaired = []
    for row in index.itertuples(index=False):
        payload = np.load(os.path.join(args.segments_dir, row.segment_id + ".npz"))
        ecg = payload["ecg"]
        pcg = payload["pcg"]
        fs = int(payload["fs"])

        ecg_frac = nan_fraction(ecg)
        pcg_frac = nan_fraction(pcg)
        if ecg_frac > max_nan_fraction or pcg_frac > max_nan_fraction:
            dropped.append(
                {
                    "segment_id": row.segment_id,
                    "record_id": row.record_id,
                    "reason": "nan_fraction_above_threshold",
                    "ecg_nan_fraction": round(ecg_frac, 5),
                    "pcg_nan_fraction": round(pcg_frac, 5),
                }
            )
            continue

        if ecg_frac > 0 or pcg_frac > 0:
            ecg = interpolate_non_finite(ecg)
            pcg = interpolate_non_finite(pcg)
            repaired.append(
                {
                    "segment_id": row.segment_id,
                    "record_id": row.record_id,
                    "ecg_nan_fraction": round(ecg_frac, 5),
                    "pcg_nan_fraction": round(pcg_frac, 5),
                }
            )

        if not np.isfinite(ecg).all() or not np.isfinite(pcg).all():
            dropped.append(
                {
                    "segment_id": row.segment_id,
                    "record_id": row.record_id,
                    "reason": "non_finite_after_interpolation",
                    "ecg_nan_fraction": round(ecg_frac, 5),
                    "pcg_nan_fraction": round(pcg_frac, 5),
                }
            )
            continue

        np.savez_compressed(
            os.path.join(args.output_dir, row.segment_id + ".npz"),
            ecg=ecg,
            pcg=pcg,
            fs=np.int32(fs),
        )
        kept.append(row._asdict())

    kept_frame = pd.DataFrame(kept)
    dropped_frame = pd.DataFrame(dropped)
    repaired_frame = pd.DataFrame(repaired)

    kept_frame.to_csv(args.index_out, index=False)
    out_dir = os.path.dirname(args.index_out)
    dropped_path = os.path.join(out_dir, DROPPED_LOG_NAME)
    repaired_path = os.path.join(out_dir, REPAIRED_LOG_NAME)
    dropped_frame.to_csv(dropped_path, index=False)
    repaired_frame.to_csv(repaired_path, index=False)

    print(f"segments kept: {len(kept_frame)}  dropped: {len(dropped_frame)}  repaired by interpolation: {len(repaired_frame)}")
    if len(repaired_frame):
        print(f"  repaired records: {repaired_frame['record_id'].nunique()}")
        print(f"  worst ecg nan fraction repaired: {repaired_frame['ecg_nan_fraction'].max()}")
        print(f"  worst pcg nan fraction repaired: {repaired_frame['pcg_nan_fraction'].max()}")
    if len(dropped_frame):
        affected = dropped_frame["record_id"].value_counts()
        print(f"  records affected by drops: {len(affected)}")
        for record_id, count in affected.items():
            print(f"    {record_id}: {count} segments dropped")

    if len(kept_frame) == 0:
        raise SystemExit("QC FAIL: segment QC dropped every segment")

    print(f"records still represented: {kept_frame['record_id'].nunique()}")
    print(f"segment label balance: normal={int((kept_frame['label'] == 0).sum())} abnormal={int((kept_frame['label'] == 1).sum())}")
    print(f"wrote {args.index_out}, {dropped_path} and {repaired_path}")


if __name__ == "__main__":
    main()
