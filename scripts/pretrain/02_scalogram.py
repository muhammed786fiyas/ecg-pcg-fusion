"""PCG scalograms for the pretraining windows -> one uint8 memmap.

The transform is Training-A's, byte for byte (scripts/features/01_scalogram.py,
duplicated by convention; tests/test_pretrain_data.py checks the two copies
agree): reflect-pad, CWT with method="fft", magnitude, crop, antialiased resize,
per-image uint8. PCG only - subsets B-F carry no ECG.

The one unavoidable difference from Training-A is the windowing. Training-A's
windows are centred on ECG R-peaks; these recordings have no ECG, so windows
are fixed, non-overlapping 3 s spans from the start of each record, capped at
pretrain.max_windows_per_record so a few long recordings cannot dominate. A 3 s
window holds roughly three beats either way, so the PCG content is comparable,
but the pretrained branch has seen un-centred windows - a stated limitation.

Windows then pass Training-A's segment QC (scripts/data/04_segment_qc.py):
dropped above max_nan_fraction non-finite, otherwise repaired by linear
interpolation.
"""

import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
import pywt
import yaml
from PIL import Image

UINT8_MAX = 255.0
CHUNK_ROWS = 200
# Must equal PAD_SCALE_FACTOR in scripts/features/01_scalogram.py.
PAD_SCALE_FACTOR = 4
PCG_MEMMAP_NAME = "pcg.uint8.npy"
ROW_INDEX_NAME = "row_index.csv"
DROPPED_LOG_NAME = "pretrain_windows_dropped.csv"


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def resize_to_square(field, size):
    return np.asarray(Image.fromarray(field, mode="F").resize((size, size), Image.BILINEAR))


def scalogram_to_uint8(field):
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low <= 0:
        return np.zeros(field.shape, dtype=np.uint8)
    scaled = (field - low) / (high - low)
    return np.clip(scaled * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)


def pad_width_for(scales, n_samples):
    return int(min(n_samples, PAD_SCALE_FACTOR * int(np.max(scales))))


def compute_scalogram(signal, scales, wavelet, fs, size):
    """Identical to scripts/features/01_scalogram.py's compute_scalogram."""
    pad = pad_width_for(scales, len(signal))
    padded = signal
    if pad > 0:
        padded = np.pad(signal, pad, mode="reflect")
    coeffs = pywt.cwt(padded, scales, wavelet, sampling_period=1.0 / fs, method="fft")[0]
    magnitude = np.abs(coeffs).astype(np.float32)
    if pad > 0:
        magnitude = magnitude[:, pad : pad + len(signal)]
    return scalogram_to_uint8(resize_to_square(magnitude, size))


def nan_fraction(signal):
    if len(signal) == 0:
        return 1.0
    return float(np.mean(~np.isfinite(signal)))


def interpolate_non_finite(signal):
    """Linear interpolation over non-finite samples, as Training-A's segment QC."""
    result = np.array(signal, dtype=np.float32, copy=True)
    bad = ~np.isfinite(result)
    if not bad.any() or not (~bad).any():
        return result
    positions = np.arange(len(result))
    result[bad] = np.interp(positions[bad], positions[~bad], result[~bad]).astype(np.float32)
    return result


def window_bounds(n_samples, window_len, max_windows):
    """Non-overlapping windows from the start of the record, the tail dropped."""
    count = min(n_samples // window_len, max_windows)
    return [(k * window_len, (k + 1) * window_len) for k in range(count)]


def list_windows(index, records_dir, window_len, max_windows, max_nan_fraction):
    """Every QC-passing window, in record order. Loads each record once."""
    rows = []
    dropped = []
    for record in index.itertuples(index=False):
        pcg = np.load(os.path.join(records_dir, record.record_id + ".npz"))["pcg"]
        for order, bounds in enumerate(window_bounds(len(pcg), window_len, max_windows)):
            start, stop = bounds
            window_id = record.record_id + "_win" + str(order).zfill(3)
            fraction = nan_fraction(pcg[start:stop])
            if fraction > max_nan_fraction:
                dropped.append({"segment_variant_id": window_id, "record_id": record.record_id,
                                "pcg_nan_fraction": round(fraction, 5)})
                continue
            rows.append({"segment_variant_id": window_id, "record_id": record.record_id,
                         "subset": record.subset, "label": int(record.label), "start": start, "stop": stop})
    return pd.DataFrame(rows), pd.DataFrame(dropped)


def transform_chunk(task):
    """Fill one contiguous row range of the memmap. Runs in a worker process."""
    memmap = np.lib.format.open_memmap(task["pcg_path"], mode="r+")
    cache = {}
    for offset, window in enumerate(task["windows"]):
        record_id, start, stop = window
        if record_id not in cache:
            cache[record_id] = np.load(os.path.join(task["records_dir"], record_id + ".npz"))["pcg"]
        segment = interpolate_non_finite(cache[record_id][start:stop])
        memmap[task["start_row"] + offset] = compute_scalogram(
            segment, task["scales"], task["wavelet"], task["fs"], task["size"]
        )
    memmap.flush()
    return len(task["windows"])


def main():
    parser = argparse.ArgumentParser(description="pretraining PCG windows -> uint8 scalogram memmap")
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--output-dir", required=True, help="data/processed/scalograms/pcg_pretrain")
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count()")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    feature_params = params["features"]["scalogram"]
    size = feature_params["image_size"]
    wavelet = feature_params["pcg_wavelet"]
    scales = np.arange(feature_params["pcg_scale_start"], feature_params["pcg_scale_stop"])
    fs = params["data"]["convert"]["expected_fs"]
    window_len = int(params["pretrain"]["window_s"] * fs)
    max_windows = params["pretrain"]["max_windows_per_record"]
    max_nan_fraction = params["data"]["segment_qc"]["max_nan_fraction"]
    workers = args.workers
    if workers <= 0:
        workers = os.cpu_count()

    print("=== pretrain 02_scalogram ===")
    print(f"pcg_wavelet={wavelet} scales {scales[0]}..{scales[-1]} window={window_len} samples "
          f"max_windows_per_record={max_windows} workers={workers}")

    index = pd.read_csv(args.index_in)
    windows, dropped = list_windows(index, args.records_dir, window_len, max_windows, max_nan_fraction)
    if len(windows) == 0:
        raise SystemExit("QC FAIL: no pretraining windows survived")
    per_record = windows.groupby("record_id").size()
    print(f"windows: {len(windows)} from {len(per_record)} records "
          f"(per record min {int(per_record.min())}, median {int(per_record.median())}, max {int(per_record.max())}); "
          f"dropped by NaN QC: {len(dropped)}")
    print(f"window label balance: normal={int((windows['label'] == 0).sum())} abnormal={int((windows['label'] == 1).sum())}")

    os.makedirs(args.output_dir, exist_ok=True)
    pcg_path = os.path.join(args.output_dir, PCG_MEMMAP_NAME)
    np.lib.format.open_memmap(pcg_path, mode="w+", dtype=np.uint8, shape=(len(windows), size, size)).flush()

    triples = list(zip(windows["record_id"], windows["start"], windows["stop"], strict=True))
    tasks = []
    for start_row in range(0, len(triples), CHUNK_ROWS):
        tasks.append({"pcg_path": pcg_path, "records_dir": args.records_dir, "start_row": start_row,
                      "windows": triples[start_row:start_row + CHUNK_ROWS], "scales": scales,
                      "wavelet": wavelet, "fs": fs, "size": size})

    done = 0
    pool = multiprocessing.Pool(processes=workers)
    for count in pool.imap_unordered(transform_chunk, tasks):
        done = done + count
        if done % 2000 < CHUNK_ROWS:
            print(f"  {done}/{len(windows)} windows transformed")
    pool.close()
    pool.join()
    if done != len(windows):
        raise SystemExit(f"QC FAIL: workers reported {done} rows but {len(windows)} windows were listed")

    row_index = windows[["segment_variant_id", "record_id", "subset", "label"]].copy()
    row_index["row_index"] = np.arange(len(windows))
    row_index.to_csv(os.path.join(args.output_dir, ROW_INDEX_NAME), index=False)
    dropped.to_csv(os.path.join(args.output_dir, DROPPED_LOG_NAME), index=False)

    written = np.lib.format.open_memmap(pcg_path, mode="r")
    sample_rows = np.linspace(0, len(windows) - 1, min(50, len(windows))).astype(int)
    blank = int(sum([1 for r in sample_rows if written[r].max() == written[r].min()]))
    print(f"sanity check: {blank} of {len(sample_rows)} sampled scalograms are constant")
    if blank > 0:
        raise SystemExit(f"QC FAIL: {blank} sampled scalograms are constant")
    print(f"wrote {pcg_path} ({round(os.path.getsize(pcg_path) / 1048576.0, 1)} MB) and {ROW_INDEX_NAME}")


if __name__ == "__main__":
    main()
