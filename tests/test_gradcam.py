"""Grad-CAM batching and hook placement, on synthetic tensors.

scripts/evaluation/02_gradcam.py profiles every test segment in batches. That
is only valid if a batched map equals the map computed for the segment on its
own - true in eval mode, where each segment's logit depends only on its own
input - so it is pinned here. Also pinned: where the hook lands, since the ResNet
branch has no `features` stack and hooking the wrong layer would still run.
No data needed, so this runs in CI.
"""

import importlib.util
import os

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 3
IMAGE_SIZE = 224


def load(path_parts, name):
    path = os.path.join(REPO_ROOT, *path_parts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_params():
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        return yaml.safe_load(handle)


def gradcam_module():
    return load(["scripts", "evaluation", "02_gradcam.py"], "gradcam")


def build(family):
    module = load(["scripts", "modeling", family, "01_train.py"], "train_" + family)
    torch.manual_seed(0)
    model = module.build_model(load_params())
    model.eval()
    return model


def test_batched_maps_equal_single_segment_maps():
    gradcam = gradcam_module()
    model = build("cross_attn_fusion")
    hooks = {
        "ecg": gradcam.GradCam(gradcam.last_conv_layer(model.ecg_branch)),
        "pcg": gradcam.GradCam(gradcam.last_conv_layer(model.pcg_branch)),
    }
    torch.manual_seed(1)
    ecg = torch.rand(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)
    pcg = torch.rand(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)

    batched_ecg, batched_pcg, _ = gradcam.batch_cams(model, hooks, ecg, pcg)
    for index in range(BATCH):
        single_ecg, single_pcg, _ = gradcam.batch_cams(
            model, hooks, ecg[index:index + 1], pcg[index:index + 1]
        )
        assert np.allclose(batched_ecg[index], single_ecg[0], atol=1e-5), f"ECG map {index} differs when batched"
        assert np.allclose(batched_pcg[index], single_pcg[0], atol=1e-5), f"PCG map {index} differs when batched"


def test_hook_lands_on_the_last_conv_of_each_branch_type():
    gradcam = gradcam_module()

    custom = build("cross_attn_fusion")
    layer = gradcam.last_conv_layer(custom.ecg_branch)
    assert layer.out_channels == 256 and layer.kernel_size == (3, 3)

    resnet = build("cross_attn_resnet18")
    layer = gradcam.last_conv_layer(resnet.ecg_branch)
    assert layer is resnet.ecg_branch.project, "ResNet hook should sit on the 1x1 projection"


def test_frequency_axis_runs_high_to_low():
    """Row 0 is the smallest scale, i.e. the HIGHEST frequency - the labelling
    bug that once inverted the ECG conclusion."""
    gradcam = gradcam_module()
    params = load_params()["features"]["scalogram"]
    scales = np.arange(params["ecg_scale_start"], params["ecg_scale_stop"])
    frequencies = gradcam.row_frequencies_hz(scales, params["ecg_wavelet"], 2000, IMAGE_SIZE)
    assert frequencies[0] > frequencies[-1]
    assert round(float(frequencies[0])) == 100 and round(float(frequencies[-1])) == 4


def test_a_uniform_map_already_puts_most_ecg_mass_below_10_hz():
    """The rows are evenly spaced in scale, so the axis is hyperbolic in Hz:
    scales 200-500 of 20-500 are the 4-10 Hz band, 62.5% of the rows. Every band
    share has to be read against this, not against zero - day 1 read it
    against zero."""
    gradcam = gradcam_module()
    params = load_params()["features"]["scalogram"]
    scales = np.arange(params["ecg_scale_start"], params["ecg_scale_stop"])
    frequencies = gradcam.row_frequencies_hz(scales, params["ecg_wavelet"], 2000, IMAGE_SIZE)
    profile = gradcam.cam_mass_profile(np.ones((IMAGE_SIZE, IMAGE_SIZE)), frequencies)
    assert abs(profile["mass_0_10hz"] - 0.625) < 0.01, profile["mass_0_10hz"]


def test_an_all_zero_map_is_flagged_empty():
    gradcam = gradcam_module()
    frequencies = np.linspace(100, 4, IMAGE_SIZE)
    profile = gradcam.cam_mass_profile(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), frequencies)
    assert profile["cam_empty"]
