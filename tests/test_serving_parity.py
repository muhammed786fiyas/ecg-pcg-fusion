"""Serving must feed the model EXACTLY the scalograms it was trained on.

The serving code duplicates the scalogram transform (self-contained by
convention), which means it can drift from the training pipeline without any
error. It did: the CWT boundary-artifact fix (reflect-padding) went into
scripts/features/01_scalogram.py, and the two serving copies were never
updated. The deployed model was fed artifact-ridden ECG scalograms - mean
|diff| ~59/255 per pixel against the training memmap - and the container smoke
test still passed, because the segments it tried were strongly abnormal.

This compares the serving transform against the stored training memmap for real
segments. It needs the pipeline outputs, so it skips in CI; run it locally
before any deploy.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALOGRAM_DIR = os.path.join(REPO_ROOT, "data", "processed", "scalograms", "default")
AUGMENTED_DIR = os.path.join(REPO_ROOT, "data", "interim", "augmented")
SERVING_FILES = ["app.py", "gradio_demo.py"]
SEGMENTS = ["a0001_seg000_orig", "a0200_seg003_orig", "a0405_seg000_orig"]
# The uint8 round trip is deterministic, so anything above one level of
# rounding is a real divergence rather than float noise.
MAX_LEVEL_DIFF = 1


def load_serving(name):
    path = os.path.join(REPO_ROOT, "scripts", "serving", name)
    spec = importlib.util.spec_from_file_location("serving_" + name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_rows():
    row_index_path = os.path.join(SCALOGRAM_DIR, "row_index.csv")
    if not os.path.exists(row_index_path):
        pytest.skip("training scalograms not present")
    rows = pd.read_csv(row_index_path)
    return dict(zip(rows["segment_variant_id"], rows["row_index"], strict=True))


def to_levels(unit_image):
    return np.round(np.asarray(unit_image) * 255.0).astype(int)


@pytest.mark.parametrize("serving_file", SERVING_FILES)
@pytest.mark.parametrize("segment_id", SEGMENTS)
def test_serving_scalogram_matches_training_memmap(serving_file, segment_id):
    row_of = training_rows()
    segment_path = os.path.join(AUGMENTED_DIR, segment_id + ".npz")
    if segment_id not in row_of or not os.path.exists(segment_path):
        pytest.skip("segment not present")

    module = load_serving(serving_file)
    ecg_memmap = np.load(os.path.join(SCALOGRAM_DIR, "ecg.uint8.npy"), mmap_mode="r")
    pcg_memmap = np.load(os.path.join(SCALOGRAM_DIR, "pcg.uint8.npy"), mmap_mode="r")
    row = int(row_of[segment_id])
    segment = np.load(segment_path)

    ecg = to_levels(module.compute_scalogram(segment["ecg"], module.ECG_SCALES, module.ECG_WAVELET, 2000))
    pcg = to_levels(module.compute_scalogram(segment["pcg"], module.PCG_SCALES, module.PCG_WAVELET, 2000))

    ecg_diff = int(np.max(np.abs(ecg - ecg_memmap[row].astype(int))))
    pcg_diff = int(np.max(np.abs(pcg - pcg_memmap[row].astype(int))))
    assert ecg_diff <= MAX_LEVEL_DIFF, (
        f"{serving_file}: ECG scalogram differs from training by up to {ecg_diff}/255 - "
        "serving and training transforms have drifted apart"
    )
    assert pcg_diff <= MAX_LEVEL_DIFF, (
        f"{serving_file}: PCG scalogram differs from training by up to {pcg_diff}/255"
    )
