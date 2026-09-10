"""Augmentation must be byte-reproducible from its seed.

The old pipeline drew from one unseeded global RNG stream in a loop, so its
output depended on filesystem iteration order. That is a large part of why this
project was rebuilt from scratch, and it is the kind of defect that leaves no
trace until someone tries to reproduce a number.
"""

import importlib.util
import os

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "data", "06_augment.py")
SEGMENTS_DIR = os.path.join(REPO_ROOT, "data", "interim", "segments_qc")
AUGMENTED_DIR = os.path.join(REPO_ROOT, "data", "interim", "augmented")


def load_module():
    spec = importlib.util.spec_from_file_location("augment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_augment_params():
    with open(os.path.join(REPO_ROOT, "params.yaml")) as handle:
        params = yaml.safe_load(handle)
    return params["global_seed"], params["data"]["augment"]


def synthetic_segment():
    rng = np.random.default_rng(0)
    ecg = rng.normal(0.0, 1.0, 6000).astype(np.float32)
    pcg = rng.normal(0.0, 0.3, 6000).astype(np.float32)
    return ecg, pcg, 2000


@pytest.mark.parametrize("variant", ["orig", "noise", "scale", "combined"])
def test_same_seed_gives_byte_identical_output(variant):
    module = load_module()
    global_seed, aug_params = load_augment_params()
    ecg, pcg, fs = synthetic_segment()

    seed = module.segment_variant_seed(global_seed, "a0001_seg000", variant)
    first = module.make_variant(ecg, pcg, fs, variant, np.random.default_rng(seed), aug_params)
    second = module.make_variant(ecg, pcg, fs, variant, np.random.default_rng(seed), aug_params)

    assert first[0].tobytes() == second[0].tobytes(), f"{variant}: ECG differs across identical seeds"
    assert first[1].tobytes() == second[1].tobytes(), f"{variant}: PCG differs across identical seeds"


def test_orig_variant_is_an_exact_copy():
    module = load_module()
    global_seed, aug_params = load_augment_params()
    ecg, pcg, fs = synthetic_segment()
    seed = module.segment_variant_seed(global_seed, "a0001_seg000", "orig")
    result = module.make_variant(ecg, pcg, fs, "orig", np.random.default_rng(seed), aug_params)
    assert result[0].tobytes() == ecg.tobytes()
    assert result[1].tobytes() == pcg.tobytes()


def test_different_segments_get_different_seeds():
    module = load_module()
    global_seed, _ = load_augment_params()
    seeds = [module.segment_variant_seed(global_seed, "a0001_seg" + str(i).zfill(3), "noise") for i in range(50)]
    assert len(set(seeds)) == len(seeds), "segment seeds collided"


def test_different_variants_of_one_segment_get_different_seeds():
    """Each variant draws from its own stream, so adding or reordering variants
    cannot silently change the values of the existing ones."""
    module = load_module()
    global_seed, _ = load_augment_params()
    seeds = [module.segment_variant_seed(global_seed, "a0001_seg000", v) for v in ["orig", "noise", "scale", "combined"]]
    assert len(set(seeds)) == len(seeds)


def test_seed_depends_on_the_global_seed():
    module = load_module()
    assert module.segment_variant_seed(42, "a0001_seg000", "noise") != module.segment_variant_seed(43, "a0001_seg000", "noise")


def test_augmented_variants_actually_differ_from_the_source():
    """A silently broken augmentation that returns the input would pass every
    reproducibility check while doing nothing."""
    module = load_module()
    global_seed, aug_params = load_augment_params()
    ecg, pcg, fs = synthetic_segment()
    for variant in ["noise", "scale", "combined"]:
        seed = module.segment_variant_seed(global_seed, "a0001_seg000", variant)
        result = module.make_variant(ecg, pcg, fs, variant, np.random.default_rng(seed), aug_params)
        assert result[0].tobytes() != ecg.tobytes(), f"{variant} left the ECG unchanged"
        assert result[1].tobytes() != pcg.tobytes(), f"{variant} left the PCG unchanged"


def test_shift_preserves_length_and_stays_finite():
    module = load_module()
    signal = np.arange(100, dtype=np.float32)
    for shift in [-100, -7, 0, 7, 99]:
        shifted = module.shift_signal(signal, shift)
        assert len(shifted) == len(signal)
        assert np.isfinite(shifted).all()


def test_shift_does_not_wrap_around():
    """np.roll would splice the tail of the window onto its front. Edge padding
    is the whole point of the custom shift."""
    module = load_module()
    signal = np.arange(100, dtype=np.float32)
    shifted = module.shift_signal(signal, 10)
    assert (shifted[:10] == signal[0]).all(), "leading samples should be edge-padded"
    assert shifted[-1] == signal[89]


def test_stored_augmentation_matches_a_fresh_recomputation():
    """The strongest form: what is on disk is what the seed produces today."""
    module = load_module()
    global_seed, aug_params = load_augment_params()
    segment_id = "a0001_seg000"
    source_path = os.path.join(SEGMENTS_DIR, segment_id + ".npz")
    if not os.path.exists(source_path):
        pytest.skip("pipeline outputs not present")

    payload = np.load(source_path)
    ecg, pcg, fs = payload["ecg"], payload["pcg"], int(payload["fs"])

    for variant in aug_params["variants"]:
        stored_path = os.path.join(AUGMENTED_DIR, segment_id + "_" + variant + ".npz")
        if not os.path.exists(stored_path):
            pytest.skip("augmented outputs not present")
        stored = np.load(stored_path)
        seed = module.segment_variant_seed(global_seed, segment_id, variant)
        fresh = module.make_variant(ecg, pcg, fs, variant, np.random.default_rng(seed), aug_params)
        assert fresh[0].tobytes() == stored["ecg"].tobytes(), f"{variant}: stored ECG does not match recomputation"
        assert fresh[1].tobytes() == stored["pcg"].tobytes(), f"{variant}: stored PCG does not match recomputation"
