"""R-peak-centred, non-overlapping 3-second windows.

R-peaks come from nk.ecg_peaks (Pan-Tompkins). A window is accepted only if it
fits entirely inside the record and does not overlap the previously accepted
window, so the segments of a record are a deterministic function of that record
alone - no RNG anywhere in this stage.

That determinism is what makes it safe to run segmentation BEFORE fold
assignment: record a0014's segments are byte-identical whichever fold it later
lands in, and assigning folds afterwards means fold sizes reflect the records
that actually survive QC.
"""

import argparse
import os

import neurokit2 as nk
import numpy as np
import pandas as pd
import yaml

SEGMENTS_INDEX_NAME = "segments_index.csv"
DROPPED_LOG_NAME = "segment_records_dropped.csv"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def detect_r_peaks(ecg, fs, method):
    """Return R-peak sample indices, or an empty array if detection fails."""
    clean = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
    except (ValueError, IndexError, ZeroDivisionError) as err:
        print(f"  r-peak detection raised {type(err).__name__}: {err}")
        return np.array([], dtype=int)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    return peaks[np.isfinite(peaks)]


def pick_windows(peaks, n_samples, window_len):
    """Non-overlapping windows centred on R-peaks, in order, skipping edge runoff."""
    half = window_len // 2
    windows = []
    next_free = 0
    for peak in peaks:
        start = int(peak) - half
        stop = start + window_len
        if start < next_free:
            continue
        if start < 0 or stop > n_samples:
            continue
        windows.append((start, stop))
        next_free = stop
    return windows


def main():
    parser = argparse.ArgumentParser(description="R-peak-centred 3s segmentation")
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    seg = params["data"]["segment"]
    window_s = seg["window_s"]
    peak_method = seg["ecg_peak_method"]

    print("=== 03_segment ===")
    print(f"window_s={window_s} ecg_peak_method={peak_method}")

    index = pd.read_csv(args.index_in)
    print(f"records in: {len(index)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    rows = []
    dropped = []
    peaks_per_record = []
    for row in index.itertuples(index=False):
        payload = np.load(os.path.join(args.records_dir, row.record_id + ".npz"))
        ecg = payload["ecg"]
        pcg = payload["pcg"]
        fs = int(payload["fs"])
        window_len = int(window_s * fs)

        peaks = detect_r_peaks(ecg, fs, peak_method)
        if len(peaks) == 0:
            dropped.append({"record_id": row.record_id, "label": int(row.label), "reason": "no_r_peaks"})
            continue
        peaks_per_record.append(len(peaks))

        windows = pick_windows(peaks, len(ecg), window_len)
        if len(windows) == 0:
            dropped.append({"record_id": row.record_id, "label": int(row.label), "reason": "no_full_windows"})
            continue

        for order, bounds in enumerate(windows):
            start, stop = bounds
            segment_id = row.record_id + "_seg" + str(order).zfill(3)
            np.savez_compressed(
                os.path.join(args.output_dir, segment_id + ".npz"),
                ecg=ecg[start:stop].astype(np.float32),
                pcg=pcg[start:stop].astype(np.float32),
                fs=np.int32(fs),
            )
            rows.append(
                {
                    "segment_id": segment_id,
                    "record_id": row.record_id,
                    "label": int(row.label),
                }
            )

    segments = pd.DataFrame(rows)
    dropped_frame = pd.DataFrame(dropped)
    segments.to_csv(args.index_out, index=False)
    dropped_path = os.path.join(os.path.dirname(args.index_out), DROPPED_LOG_NAME)
    dropped_frame.to_csv(dropped_path, index=False)

    surviving_records = segments["record_id"].nunique() if len(segments) else 0
    print(f"records with segments: {surviving_records}  records dropped: {len(dropped_frame)}")
    if len(dropped_frame):
        for reason, count in dropped_frame["reason"].value_counts().items():
            print(f"  drop reason {reason}: {count}")
        print(f"  dropped ids: {sorted(dropped_frame['record_id'])}")
    if len(peaks_per_record):
        print(f"r-peaks per record: min={min(peaks_per_record)} median={int(np.median(peaks_per_record))} max={max(peaks_per_record)}")
    print(f"segments written: {len(segments)}")
    if len(segments):
        per_record = segments.groupby("record_id").size()
        print(f"segments per record: min={int(per_record.min())} median={int(per_record.median())} max={int(per_record.max())} mean={round(float(per_record.mean()), 2)}")
        print(f"segment label balance: normal={int((segments['label'] == 0).sum())} abnormal={int((segments['label'] == 1).sum())}")

    if len(segments) == 0:
        raise SystemExit("QC FAIL: segmentation produced zero segments")

    print(f"wrote {args.index_out} and {dropped_path}")


if __name__ == "__main__":
    main()
