"""Guards for the PCG pretraining data pipeline (scripts/pretrain/).

Pretraining is only legitimate if it can never touch a Training-A record - the
CV data - and if the pretrained branch sees the same image transform the
fine-tuned model is trained and tested on. Both are pinned here, on synthetic
inputs, so this runs in CI.
"""

import importlib.util
import inspect
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path_parts, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, *path_parts))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_module():
    return load(["scripts", "pretrain", "01_convert.py"], "pretrain_convert")


def scalogram_module():
    return load(["scripts", "pretrain", "02_scalogram.py"], "pretrain_scalogram")


def split_module():
    return load(["scripts", "pretrain", "03_split.py"], "pretrain_split")


def test_training_a_records_are_rejected():
    module = convert_module()
    module.reject_training_a(["b0001", "e00123", "f0045"])
    with pytest.raises(SystemExit):
        module.reject_training_a(["b0001", "a0007"])


def test_split_sees_only_record_ids_strata_and_a_seed():
    """The same structural contract as the Training-A split."""
    parameters = list(inspect.signature(split_module().split_records).parameters)
    assert parameters == ["record_ids", "strata", "seed", "val_frac"]


def test_split_is_record_level_deterministic_and_order_free():
    module = split_module()
    record_ids = ["b" + str(i).zfill(4) for i in range(40)] + ["e" + str(i).zfill(4) for i in range(60)]
    strata = ["b_0"] * 20 + ["b_1"] * 20 + ["e_0"] * 40 + ["e_1"] * 20
    first = module.split_records(record_ids, strata, 42, 0.15)
    shuffled = list(zip(record_ids, strata, strict=True))[::-1]
    second = module.split_records([pair[0] for pair in shuffled], [pair[1] for pair in shuffled], 42, 0.15)
    assert first == second, "the split depends on input order"
    assert 10 <= len(first) <= 20


def test_windows_are_non_overlapping_in_bounds_and_capped():
    module = scalogram_module()
    windows = module.window_bounds(n_samples=6000 * 30 + 1234, window_len=6000, max_windows=20)
    assert len(windows) == 20
    for index in range(len(windows) - 1):
        assert windows[index][1] <= windows[index + 1][0]
    assert windows[-1][1] <= 6000 * 30 + 1234
    assert module.window_bounds(n_samples=5999, window_len=6000, max_windows=20) == []


def test_pretraining_transform_matches_training_a_transform():
    """The pretrained branch must see exactly the images it is fine-tuned on."""
    features = load(["scripts", "features", "01_scalogram.py"], "features_scalogram")
    pretrain = scalogram_module()
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        params = yaml.safe_load(handle)["features"]["scalogram"]
    scales = np.arange(params["pcg_scale_start"], params["pcg_scale_stop"])
    rng = np.random.default_rng(0)
    signal = rng.normal(size=6000).astype(np.float32)
    expected = features.compute_scalogram(signal, scales, params["pcg_wavelet"], 2000, params["image_size"])
    actual = pretrain.compute_scalogram(signal, scales, params["pcg_wavelet"], 2000, params["image_size"])
    assert np.array_equal(expected, actual)


def load_params():
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        return yaml.safe_load(handle)


def test_fine_tuning_loads_exactly_the_pretrained_pcg_branch(tmp_path):
    """The PCG branch must come from the checkpoint; the ECG branch must not
    be touched - otherwise the pair no longer isolates PCG pretraining."""
    pretrain = load(["scripts", "modeling", "pcg_pretrain", "01_train.py"], "train_pcg_pretrain")
    finetune = load(["scripts", "modeling", "cross_attn_resnet18_pcgpre", "01_train.py"], "train_ca_pcgpre")
    params = load_params()

    torch.manual_seed(0)
    source = pretrain.build_model(params)
    path = tmp_path / "pretrain_best.pth"
    torch.save({"model": source.state_dict(), "val_auc": 0.9}, path)

    torch.manual_seed(1)
    model = finetune.build_model(params)
    ecg_before = {key: value.clone() for key, value in model.ecg_branch.state_dict().items()}
    finetune.load_pcg_init(model, str(path))

    loaded = model.pcg_branch.state_dict()
    for key, value in source.pcg_branch.state_dict().items():
        assert torch.equal(loaded[key], value), f"pcg_branch.{key} was not loaded"
    for key, value in model.ecg_branch.state_dict().items():
        assert torch.equal(value, ecg_before[key]), f"ecg_branch.{key} changed"


def test_a_missing_pretraining_checkpoint_is_fatal(tmp_path):
    """No silent fallback to ImageNet weights under the pretrained name."""
    finetune = load(["scripts", "modeling", "resnet18_pcg_only_pcgpre", "01_train.py"], "train_pcg_pcgpre")
    model = finetune.build_model(load_params())
    with pytest.raises(SystemExit):
        finetune.load_pcg_init(model, str(tmp_path / "missing.pth"))


def test_nan_windows_are_dropped_and_small_gaps_repaired(tmp_path):
    module = scalogram_module()
    pcg = np.ones(12000, dtype=np.float32)
    pcg[:5500] = np.nan          # first window 92% missing -> dropped
    pcg[7000:7010] = np.nan      # second window: a small gap -> kept
    np.savez_compressed(tmp_path / "b0001.npz", pcg=pcg, fs=np.int32(2000))
    index = pd.DataFrame({"record_id": ["b0001"], "subset": ["training-b"], "label": [1]})
    windows, dropped = module.list_windows(index, str(tmp_path), 6000, 20, 0.9)
    assert list(windows["segment_variant_id"]) == ["b0001_win001"]
    assert list(dropped["segment_variant_id"]) == ["b0001_win000"]
    repaired = module.interpolate_non_finite(pcg[6000:12000])
    assert np.isfinite(repaired).all()
