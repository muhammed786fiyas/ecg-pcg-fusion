"""Dataset contract: tensor shapes, label range, no NaN in a batch."""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_ROOT = os.path.join(REPO_ROOT, "data", "processed", "manifests")
SCALOGRAM_DIR = os.path.join(REPO_ROOT, "data", "processed", "scalograms", "default")
IMAGE_SIZE = 224


def load_dataset_class():
    path = os.path.join(REPO_ROOT, "scripts", "modeling", "dual_cnn", "01_train.py")
    spec = importlib.util.spec_from_file_location("train_dual_cnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synthetic_scalograms(tmp_path, n_rows):
    """A miniature stand-in so the contract is testable without the real data."""
    ecg = np.random.randint(0, 256, (n_rows, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    pcg = np.random.randint(0, 256, (n_rows, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    np.save(os.path.join(tmp_path, "ecg.uint8.npy"), ecg)
    np.save(os.path.join(tmp_path, "pcg.uint8.npy"), pcg)

    manifest = pd.DataFrame(
        {
            "row_index": np.arange(n_rows),
            "segment_variant_id": ["a0001_seg" + str(i).zfill(3) + "_orig" for i in range(n_rows)],
            "record_id": ["a0001"] * n_rows,
            "label": [i % 2 for i in range(n_rows)],
        }
    )
    manifest_path = os.path.join(tmp_path, "train.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def test_synthetic_item_shapes_and_range(tmp_path):
    module = load_dataset_class()
    manifest_path = make_synthetic_scalograms(str(tmp_path), 8)
    dataset = module.ScalogramDataset(manifest_path, str(tmp_path))

    assert len(dataset) == 8
    ecg, pcg, label = dataset[0]
    assert tuple(ecg.shape) == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(pcg.shape) == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert ecg.dtype == torch.float32
    assert float(ecg.min()) >= 0.0 and float(ecg.max()) <= 1.0, "uint8 should be scaled to [0, 1]"
    assert float(label) in (0.0, 1.0)


def test_synthetic_batch_has_no_nan(tmp_path):
    module = load_dataset_class()
    manifest_path = make_synthetic_scalograms(str(tmp_path), 16)
    dataset = module.ScalogramDataset(manifest_path, str(tmp_path))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)

    for ecg, pcg, labels in loader:
        assert torch.isfinite(ecg).all()
        assert torch.isfinite(pcg).all()
        assert torch.isfinite(labels).all()
        assert set(labels.unique().tolist()).issubset({0.0, 1.0})


def test_dataset_reads_the_row_the_manifest_names(tmp_path):
    """The manifest's row_index is the only membership authority. If the Dataset
    ever enumerated the memmap instead, val rows would silently become train rows."""
    module = load_dataset_class()
    n_rows = 10
    ecg = np.zeros((n_rows, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    pcg = np.zeros((n_rows, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    ecg[7] = 255
    np.save(os.path.join(str(tmp_path), "ecg.uint8.npy"), ecg)
    np.save(os.path.join(str(tmp_path), "pcg.uint8.npy"), pcg)

    manifest = pd.DataFrame(
        {
            "row_index": [7],
            "segment_variant_id": ["a0001_seg007_orig"],
            "record_id": ["a0001"],
            "label": [1],
        }
    )
    manifest_path = os.path.join(str(tmp_path), "one.csv")
    manifest.to_csv(manifest_path, index=False)

    dataset = module.ScalogramDataset(manifest_path, str(tmp_path))
    ecg_tensor = dataset[0][0]
    assert float(ecg_tensor.min()) == 1.0, "dataset did not read row 7"


@pytest.mark.parametrize("tag", ["dev", "cv_fold0"])
def test_real_manifest_batch_is_clean(tag):
    module = load_dataset_class()
    manifest_path = os.path.join(MANIFEST_ROOT, tag, "test.csv")
    if not os.path.exists(manifest_path) or not os.path.isdir(SCALOGRAM_DIR):
        pytest.skip("pipeline outputs not present")

    dataset = module.ScalogramDataset(manifest_path, SCALOGRAM_DIR)
    assert len(dataset) > 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0)
    ecg, pcg, labels = next(iter(loader))

    assert tuple(ecg.shape)[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(ecg).all() and torch.isfinite(pcg).all()
    assert float(ecg.max()) <= 1.0 and float(ecg.min()) >= 0.0
    assert set(labels.unique().tolist()).issubset({0.0, 1.0})


def test_pos_weight_is_computed_from_the_given_manifest_only(tmp_path):
    """pos_weight is a fitted statistic. It must come from the training manifest
    handed to it and nothing wider."""
    module = load_dataset_class()
    manifest = pd.DataFrame({"label": [1] * 30 + [0] * 10})
    path = os.path.join(str(tmp_path), "train.csv")
    manifest.to_csv(path, index=False)
    assert module.compute_pos_weight(path) == pytest.approx(10.0 / 30.0)
