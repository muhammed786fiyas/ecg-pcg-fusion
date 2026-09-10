"""Forward-pass shape contracts for every architecture, on synthetic tensors.

No data needed, so this runs in CI. It also guards the deliberate duplication of
the model definitions across training scripts (docs/logs/tasks/3-modeling.md):
if one copy of CNNBranch drifts, its family's shapes stop matching here.
"""

import importlib.util
import os

import pytest
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = [
    "dual_cnn",
    "ecg_only",
    "pcg_only",
    "cbam_fusion",
    "cross_attn_fusion",
    "cross_attn_resnet18",
    "warm_start_fusion",
]

CUSTOM_BACKBONE_FAMILIES = [
    "dual_cnn",
    "ecg_only",
    "cbam_fusion",
    "cross_attn_fusion",
    "warm_start_fusion",
]

BATCH = 2
IMAGE_SIZE = 224


def load_family(name):
    path = os.path.join(REPO_ROOT, "scripts", "modeling", name, "01_train.py")
    spec = importlib.util.spec_from_file_location("train_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_params():
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        return yaml.safe_load(handle)


def synthetic_batch():
    ecg = torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)
    pcg = torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)
    return ecg, pcg


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_returns_one_logit_per_example(family):
    module = load_family(family)
    model = module.build_model(load_params())
    model.eval()
    ecg, pcg = synthetic_batch()
    with torch.no_grad():
        output = model(ecg, pcg)
    assert output.shape == (BATCH,), f"{family} returned {tuple(output.shape)}, expected ({BATCH},)"
    assert torch.isfinite(output).all(), f"{family} produced non-finite logits"


@pytest.mark.parametrize("family", FAMILIES)
def test_model_family_constant_matches_its_folder(family):
    """Catches a copy-paste that forgot to rename MODEL_FAMILY - which would
    silently write one family's results over another's."""
    module = load_family(family)
    assert module.MODEL_FAMILY == family


@pytest.mark.parametrize("family", CUSTOM_BACKBONE_FAMILIES)
def test_custom_branch_feature_map_is_14x14x256(family):
    """The spec: 4 conv blocks on a 224x224 input give a 14x14x256 map before
    global pooling. The cross-attention block's 196 tokens depend on it, and so
    does where Grad-CAM hooks."""
    module = load_family(family)
    branch = module.CNNBranch()
    branch.eval()
    with torch.no_grad():
        maps = branch.feature_maps(torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE))
        pooled = branch(torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert tuple(maps.shape) == (BATCH, 256, 14, 14), f"{family}: {tuple(maps.shape)}"
    assert tuple(pooled.shape) == (BATCH, 256)


@pytest.mark.parametrize("family", FAMILIES)
def test_backward_pass_produces_gradients(family):
    """A model whose branches are accidentally detached would train to nothing."""
    module = load_family(family)
    model = module.build_model(load_params())
    ecg, pcg = synthetic_batch()
    output = model(ecg, pcg)
    output.sum().backward()

    with_grad = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) > 0, f"{family} produced no gradients"


def test_unimodal_models_use_only_their_own_modality():
    """ecg_only must ignore the PCG tensor, and vice versa. If a baseline
    quietly saw both modalities the whole fusion ablation would be meaningless."""
    ecg, pcg = synthetic_batch()
    other = torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)

    ecg_model = load_family("ecg_only").build_model(load_params())
    ecg_model.eval()
    with torch.no_grad():
        assert torch.allclose(ecg_model(ecg, pcg), ecg_model(ecg, other)), (
            "ecg_only changed its output when only the PCG input changed"
        )

    pcg_model = load_family("pcg_only").build_model(load_params())
    pcg_model.eval()
    with torch.no_grad():
        assert torch.allclose(pcg_model(ecg, pcg), pcg_model(other, pcg)), (
            "pcg_only changed its output when only the ECG input changed"
        )


def test_fusion_models_use_both_modalities():
    """The mirror: a fusion model that ignores one branch is a silent bug."""
    ecg, pcg = synthetic_batch()
    other = torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)

    for family in ["dual_cnn", "cbam_fusion", "cross_attn_fusion", "warm_start_fusion"]:
        model = load_family(family).build_model(load_params())
        model.eval()
        with torch.no_grad():
            base = model(ecg, pcg)
            changed_pcg = model(ecg, other)
            changed_ecg = model(other, pcg)
        assert not torch.allclose(base, changed_pcg), f"{family} ignores the PCG branch"
        assert not torch.allclose(base, changed_ecg), f"{family} ignores the ECG branch"


def test_cross_attention_produces_196_tokens():
    module = load_family("cross_attn_fusion")
    model = module.build_model(load_params())
    model.eval()
    with torch.no_grad():
        maps = model.ecg_branch.feature_maps(torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE))
        tokens = model.to_tokens(maps)
    assert tuple(tokens.shape) == (BATCH, 196, 256), f"got {tuple(tokens.shape)}"
