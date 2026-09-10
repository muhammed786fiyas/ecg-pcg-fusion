"""CWT scalograms -> uint8 memmap arrays, one config at a time.

For each (wavelet_config, modality) this writes a single memmap of shape
(N, IMAGE_SIZE, IMAGE_SIZE) plus a row-index CSV mapping segment_variant_id to
its row.

Why a uint8 memmap and not a PNG. The old pipeline rendered each scalar
magnitude field through matplotlib's jet colormap into a 3-channel PNG. That
triples the data, throws away precision through a perceptually non-uniform
colour mapping, and makes plt.savefig plus PNG decode the dominant cost of every
training epoch. A uint8 memmap loads essentially for free and keeps the
scalogram as the scalar field it actually is. jet PNGs are rendered separately,
and only for the handful of segments that appear in paper figures and Grad-CAM
overlays.

Normalization is PER IMAGE (each scalogram scaled to 0-255 by its own min/max),
so no statistic is ever pooled across segments and there is nothing here that
could leak between folds.
"""

import argparse
import os

import numpy as np
import pandas as pd
import pywt
import yaml
from scipy.ndimage import zoom

UINT8_MAX = 255.0
PROGRESS_EVERY = 1000


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def resize_to_square(field, size):
    """Bilinear resize of a 2-D array to size x size."""
    factors = (size / float(field.shape[0]), size / float(field.shape[1]))
    return zoom(field, factors, order=1)


def scalogram_to_uint8(field):
    """Scale one scalogram to 0-255 by its own min/max. Per image, so no pooling."""
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low <= 0:
        return np.zeros(field.shape, dtype=np.uint8)
    scaled = (field - low) / (high - low)
    return np.clip(scaled * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)


def compute_scalogram(signal, scales, wavelet, fs, size):
    coeffs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)[0]
    magnitude = np.abs(coeffs)
    resized = resize_to_square(magnitude, size)
    return scalogram_to_uint8(resized)


def main():
    parser = argparse.ArgumentParser(description="CWT scalograms to uint8 memmaps")
    parser.add_argument("--augmented-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--output-dir", required=True, help="data/processed/scalograms/<config>")
    parser.add_argument("--config-name", default="default", help="wavelet config name")
    parser.add_argument("--ecg-wavelet", default="", help="override params.yaml ECG wavelet")
    parser.add_argument("--pcg-wavelet", default="", help="override params.yaml PCG wavelet")
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

    print("=== 01_scalogram ===")
    print(f"config={args.config_name} image_size={size}")
    print(f"ecg_wavelet={ecg_wavelet} ecg_scales={ecg_scales[0]}..{ecg_scales[-1]} ({len(ecg_scales)} scales)")
    print(f"pcg_wavelet={pcg_wavelet} pcg_scales={pcg_scales[0]}..{pcg_scales[-1]} ({len(pcg_scales)} scales)")

    continuous = pywt.wavelist(kind="continuous")
    for name in [ecg_wavelet, pcg_wavelet]:
        base = name.split("-")[0]
        base = "".join([c for c in base if not c.isdigit() and c != "."])
        if base not in continuous and name not in continuous:
            raise SystemExit(
                f"QC FAIL: {name} is not a continuous wavelet. pywt.cwt supports only {continuous}. "
                "This is why db4 was replaced by gaus4 - see docs/logs/tasks/2-features.md"
            )

    index = pd.read_csv(args.index_in)
    n_rows = len(index)
    print(f"rows to transform: {n_rows}")
    if n_rows == 0:
        raise SystemExit("QC FAIL: augmented index is empty")

    os.makedirs(args.output_dir, exist_ok=True)
    ecg_path = os.path.join(args.output_dir, "ecg.uint8.memmap")
    pcg_path = os.path.join(args.output_dir, "pcg.uint8.memmap")
    row_index_path = os.path.join(args.output_dir, "row_index.csv")

    ecg_memmap = np.lib.format.open_memmap(
        ecg_path, mode="w+", dtype=np.uint8, shape=(n_rows, size, size)
    )
    pcg_memmap = np.lib.format.open_memmap(
        pcg_path, mode="w+", dtype=np.uint8, shape=(n_rows, size, size)
    )

    reported_freqs = False
    for position, row in enumerate(index.itertuples(index=False)):
        payload = np.load(os.path.join(args.augmented_dir, row.segment_variant_id + ".npz"))
        ecg = payload["ecg"]
        pcg = payload["pcg"]
        fs = int(payload["fs"])

        if not reported_freqs:
            ecg_freqs = pywt.scale2frequency(ecg_wavelet, ecg_scales) * fs
            pcg_freqs = pywt.scale2frequency(pcg_wavelet, pcg_scales) * fs
            print(f"measured ECG band: {round(float(ecg_freqs.min()), 2)} - {round(float(ecg_freqs.max()), 2)} Hz")
            print(f"measured PCG band: {round(float(pcg_freqs.min()), 2)} - {round(float(pcg_freqs.max()), 2)} Hz")
            reported_freqs = True

        ecg_memmap[position] = compute_scalogram(ecg, ecg_scales, ecg_wavelet, fs, size)
        pcg_memmap[position] = compute_scalogram(pcg, pcg_scales, pcg_wavelet, fs, size)

        if (position + 1) % PROGRESS_EVERY == 0:
            print(f"  {position + 1}/{n_rows} rows transformed")

    ecg_memmap.flush()
    pcg_memmap.flush()

    row_index = index.copy()
    row_index["row_index"] = np.arange(n_rows)
    row_index.to_csv(row_index_path, index=False)

    ecg_mb = round(os.path.getsize(ecg_path) / (1024.0 * 1024.0), 1)
    pcg_mb = round(os.path.getsize(pcg_path) / (1024.0 * 1024.0), 1)
    print(f"wrote {ecg_path} ({ecg_mb} MB) and {pcg_path} ({pcg_mb} MB)")
    print(f"wrote {row_index_path} with {n_rows} rows")


if __name__ == "__main__":
    main()
