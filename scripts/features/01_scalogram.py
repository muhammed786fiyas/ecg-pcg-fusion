"""CWT scalograms -> uint8 memmap arrays, one wavelet config at a time.

For each (wavelet_config, modality) this writes a single memmap of shape
(N, IMAGE_SIZE, IMAGE_SIZE) plus a row-index CSV mapping segment_variant_id to
its row.

Why a uint8 memmap and not a PNG. The old pipeline rendered each scalar
magnitude field through matplotlib's jet colormap into a 3-channel PNG. That
triples the data, throws away precision through a perceptually non-uniform
colour mapping, and makes plt.savefig plus PNG decode the dominant cost of every
training epoch. A uint8 memmap loads essentially for free and keeps the
scalogram as the scalar field it actually is. jet PNGs are rendered separately
and only for the handful of segments used in paper figures and Grad-CAM
overlays.

Normalization is PER IMAGE - each scalogram is scaled to 0-255 by its own
min/max - so no statistic is ever pooled across segments and there is nothing
here that could leak between folds.

Three performance decisions, all measured on this machine (notes in
docs/logs/tasks/2-features.md):
  - pywt.cwt(method="fft"), 4.2x faster than the default direct convolution at
    481 ECG scales and numerically identical to 2e-5 relative.
  - PIL's antialiased resize, not scipy.ndimage.zoom. Going from 6000 time
    columns to 224 is a 27x reduction; zoom interpolates at sample points and
    loses 3.1% of the field's mean, PIL area-averages and preserves it to 0.01%.
  - A process pool over the rows. The work is embarrassingly parallel and each
    worker writes a disjoint row range of the same memmap.
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
# Reflect-pad by this many times the largest scale before transforming, to keep
# the CWT cone of influence outside the window that is actually kept.
PAD_SCALE_FACTOR = 4
ECG_MEMMAP_NAME = "ecg.uint8.npy"
PCG_MEMMAP_NAME = "pcg.uint8.npy"
ROW_INDEX_NAME = "row_index.csv"


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def resize_to_square(field, size):
    """Antialiased resize of a 2-D float32 field to size x size.

    PIL's resampler scales its filter support by the reduction factor, so this
    is an area average rather than a point sample. That matters: the time axis
    is reduced ~27x here, and point sampling would discard most of it.
    """
    return np.asarray(Image.fromarray(field, mode="F").resize((size, size), Image.BILINEAR))


def scalogram_to_uint8(field):
    """Scale one scalogram to 0-255 by its own min/max. Per image, so no pooling."""
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low <= 0:
        return np.zeros(field.shape, dtype=np.uint8)
    scaled = (field - low) / (high - low)
    return np.clip(scaled * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)


def pad_width_for(scales, n_samples):
    """How far to reflect-pad before transforming, from the largest wavelet's reach.

    A wavelet at scale s has effective support of a few times s. At the ECG
    scales used here (up to 500) that is comparable to the 6000-sample window
    itself, so the cone of influence swamps the transform. PAD_SCALE_FACTOR * the
    largest scale puts the boundary far enough away that the returned window is
    all valid.
    """
    return int(min(n_samples, PAD_SCALE_FACTOR * int(np.max(scales))))


def compute_scalogram(signal, scales, wavelet, fs, size):
    """CWT magnitude of a reflect-padded signal, cropped back to the real window.

    The padding is not cosmetic. Transforming the bare 3-second window leaves a
    boundary artifact that DOMINATES the ECG image: measured on this data, the
    outer columns carried ~19x the energy of the interior, and after per-image
    min/max scaling the actual cardiac content was compressed to a mean of
    4.6/255 with 85% of pixels below 26/255. Reflect-padding and cropping raises
    the interior mean to 59.3/255 - about 13x more usable dynamic range - and
    drops the edge/interior ratio to 0.55.

    It matters twice over. The model was being handed images whose 8-bit range
    was spent on an artifact, and Grad-CAM would have highlighted that artifact,
    which would have turned a preprocessing bug into a false conclusion about
    what the model attends to physiologically.
    """
    pad = pad_width_for(scales, len(signal))
    padded = np.pad(signal, pad, mode="reflect") if pad > 0 else signal
    coeffs = pywt.cwt(padded, scales, wavelet, sampling_period=1.0 / fs, method="fft")[0]
    magnitude = np.abs(coeffs).astype(np.float32)
    if pad > 0:
        magnitude = magnitude[:, pad : pad + len(signal)]
    return scalogram_to_uint8(resize_to_square(magnitude, size))


def transform_chunk(task):
    """Fill one contiguous row range of both memmaps. Runs in a worker process."""
    augmented_dir = task["augmented_dir"]
    variant_ids = task["variant_ids"]
    start_row = task["start_row"]
    size = task["size"]

    ecg_memmap = np.lib.format.open_memmap(task["ecg_path"], mode="r+")
    pcg_memmap = np.lib.format.open_memmap(task["pcg_path"], mode="r+")

    for offset, variant_id in enumerate(variant_ids):
        payload = np.load(os.path.join(augmented_dir, variant_id + ".npz"))
        fs = int(payload["fs"])
        ecg_memmap[start_row + offset] = compute_scalogram(
            payload["ecg"], task["ecg_scales"], task["ecg_wavelet"], fs, size
        )
        pcg_memmap[start_row + offset] = compute_scalogram(
            payload["pcg"], task["pcg_scales"], task["pcg_wavelet"], fs, size
        )

    ecg_memmap.flush()
    pcg_memmap.flush()
    return len(variant_ids)


def build_tasks(index, augmented_dir, ecg_path, pcg_path, ecg_scales, pcg_scales, ecg_wavelet, pcg_wavelet, size):
    variant_ids = index["segment_variant_id"].tolist()
    tasks = []
    for start in range(0, len(variant_ids), CHUNK_ROWS):
        tasks.append(
            {
                "augmented_dir": augmented_dir,
                "ecg_path": ecg_path,
                "pcg_path": pcg_path,
                "variant_ids": variant_ids[start : start + CHUNK_ROWS],
                "start_row": start,
                "ecg_scales": ecg_scales,
                "pcg_scales": pcg_scales,
                "ecg_wavelet": ecg_wavelet,
                "pcg_wavelet": pcg_wavelet,
                "size": size,
            }
        )
    return tasks


def check_wavelet_is_continuous(name):
    """pywt.cwt only accepts continuous wavelets. This is the db4 trap."""
    continuous = pywt.wavelist(kind="continuous")
    if name in continuous:
        return
    family = "".join([c for c in name.split("-")[0] if not c.isdigit() and c != "."])
    if family in continuous:
        return
    raise SystemExit(
        f"QC FAIL: '{name}' is not a continuous wavelet, so pywt.cwt cannot use it. "
        f"Supported: {continuous}. This is exactly why db4 was replaced by gaus4 - "
        "see docs/logs/tasks/2-features.md"
    )


def main():
    parser = argparse.ArgumentParser(description="CWT scalograms to uint8 memmaps")
    parser.add_argument("--augmented-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--output-dir", required=True, help="data/processed/scalograms/<config>")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--ecg-wavelet", default="", help="override params.yaml ECG wavelet")
    parser.add_argument("--pcg-wavelet", default="", help="override params.yaml PCG wavelet")
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count()")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    feature_params = params["features"]["scalogram"]
    size = feature_params["image_size"]

    ecg_wavelet = args.ecg_wavelet if args.ecg_wavelet else feature_params["ecg_wavelet"]
    pcg_wavelet = args.pcg_wavelet if args.pcg_wavelet else feature_params["pcg_wavelet"]

    # Scale ranges are held fixed across every wavelet. Retuning scales per
    # wavelet is a different and much larger experiment, and mixing the two
    # would confound the wavelet-sensitivity contribution.
    ecg_scales = np.arange(feature_params["ecg_scale_start"], feature_params["ecg_scale_stop"])
    pcg_scales = np.arange(feature_params["pcg_scale_start"], feature_params["pcg_scale_stop"])

    workers = args.workers if args.workers > 0 else os.cpu_count()

    print("=== 01_scalogram ===")
    print(f"config={args.config_name} image_size={size} workers={workers}")
    print(f"ecg_wavelet={ecg_wavelet} scales {ecg_scales[0]}..{ecg_scales[-1]} ({len(ecg_scales)} scales)")
    print(f"pcg_wavelet={pcg_wavelet} scales {pcg_scales[0]}..{pcg_scales[-1]} ({len(pcg_scales)} scales)")

    check_wavelet_is_continuous(ecg_wavelet)
    check_wavelet_is_continuous(pcg_wavelet)

    index = pd.read_csv(args.index_in)
    n_rows = len(index)
    print(f"rows to transform: {n_rows}")
    if n_rows == 0:
        raise SystemExit("QC FAIL: augmented index is empty")
    if index["segment_variant_id"].nunique() != n_rows:
        raise SystemExit("QC FAIL: duplicate segment_variant_id in the augmented index")

    # Report the frequency band these scales actually cover. The brief's
    # annotation for ECG was wrong; the measured numbers are what go in the paper.
    sample_fs = int(np.load(os.path.join(args.augmented_dir, index["segment_variant_id"].iloc[0] + ".npz"))["fs"])
    ecg_freqs = pywt.scale2frequency(ecg_wavelet, ecg_scales) * sample_fs
    pcg_freqs = pywt.scale2frequency(pcg_wavelet, pcg_scales) * sample_fs
    print(f"measured ECG band: {round(float(ecg_freqs.min()), 2)} - {round(float(ecg_freqs.max()), 2)} Hz")
    print(f"measured PCG band: {round(float(pcg_freqs.min()), 2)} - {round(float(pcg_freqs.max()), 2)} Hz")

    os.makedirs(args.output_dir, exist_ok=True)
    ecg_path = os.path.join(args.output_dir, ECG_MEMMAP_NAME)
    pcg_path = os.path.join(args.output_dir, PCG_MEMMAP_NAME)

    # Allocate both memmaps up front so workers can write disjoint row ranges.
    np.lib.format.open_memmap(ecg_path, mode="w+", dtype=np.uint8, shape=(n_rows, size, size)).flush()
    np.lib.format.open_memmap(pcg_path, mode="w+", dtype=np.uint8, shape=(n_rows, size, size)).flush()

    tasks = build_tasks(
        index, args.augmented_dir, ecg_path, pcg_path,
        ecg_scales, pcg_scales, ecg_wavelet, pcg_wavelet, size,
    )
    print(f"dispatching {len(tasks)} chunks of up to {CHUNK_ROWS} rows")

    done = 0
    pool = multiprocessing.Pool(processes=workers)
    for count in pool.imap_unordered(transform_chunk, tasks):
        done = done + count
        print(f"  {done}/{n_rows} rows transformed")
    pool.close()
    pool.join()

    if done != n_rows:
        raise SystemExit(f"QC FAIL: workers reported {done} rows but the index has {n_rows}")

    row_index = index.copy()
    row_index["row_index"] = np.arange(n_rows)
    row_index_path = os.path.join(args.output_dir, ROW_INDEX_NAME)
    row_index.to_csv(row_index_path, index=False)

    # A blank memmap is a silent failure mode - an all-zero image trains fine and
    # means nothing. Check that the written data is actually varying.
    written = np.lib.format.open_memmap(ecg_path, mode="r")
    sample_rows = np.linspace(0, n_rows - 1, min(50, n_rows)).astype(int)
    blank = int(sum([1 for r in sample_rows if written[r].max() == written[r].min()]))
    print(f"sanity check: {blank} of {len(sample_rows)} sampled ECG scalograms are constant")
    if blank > 0:
        raise SystemExit(f"QC FAIL: {blank} sampled scalograms are constant - the transform produced blank images")

    ecg_mb = round(os.path.getsize(ecg_path) / (1024.0 * 1024.0), 1)
    pcg_mb = round(os.path.getsize(pcg_path) / (1024.0 * 1024.0), 1)
    print(f"wrote {ecg_path} ({ecg_mb} MB) and {pcg_path} ({pcg_mb} MB)")
    print(f"wrote {row_index_path} with {n_rows} rows")


if __name__ == "__main__":
    main()
