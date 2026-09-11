"""The modality permutation test's shuffling, pinned on synthetic records.

scripts/evaluation/07_modality_permutation.py measures how much a fusion model
relies on each modality by pairing every segment's ECG with the PCG of a
DIFFERENT record (and vice versa). If a segment could keep its own record's
signal, the test would understate the reliance, so the shuffle must never map a
record onto itself and must always point at a real row.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "evaluation", "07_modality_permutation.py")


def load_module():
    spec = importlib.util.spec_from_file_location("modality_permutation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_frame():
    """Five records with uneven segment counts, as real records have."""
    counts = {"a": 3, "b": 1, "c": 4, "d": 2, "e": 5}
    record_ids = []
    for record, count in counts.items():
        record_ids.extend([record] * count)
    return pd.DataFrame({"record_id": record_ids, "label": [0] * len(record_ids)})


@pytest.mark.parametrize("n", [2, 3, 10, 81])
def test_derangement_has_no_fixed_points(n):
    module = load_module()
    rng = np.random.default_rng(0)
    for _ in range(20):
        order = module.derangement(n, rng)
        assert sorted(order.tolist()) == list(range(n))
        assert not np.any(order == np.arange(n))


def test_no_segment_keeps_its_own_record():
    module = load_module()
    frame = make_frame()
    rng = np.random.default_rng(1)
    for _ in range(20):
        donors = module.donor_rows(frame, rng)
        assert donors.min() >= 0 and donors.max() < len(frame)
        own = frame["record_id"].to_numpy()
        assert not np.any(own[donors] == own), "a segment was paired with its own record"


def test_each_record_takes_all_its_segments_from_one_donor():
    module = load_module()
    frame = make_frame()
    donors = module.donor_rows(frame, np.random.default_rng(2))
    own = frame["record_id"].to_numpy()
    for record in frame["record_id"].unique():
        donor_records = set(own[donors[own == record]].tolist())
        assert len(donor_records) == 1, f"record {record} drew from {donor_records}"


def test_a_single_record_cannot_be_deranged():
    module = load_module()
    with pytest.raises(SystemExit):
        module.derangement(1, np.random.default_rng(0))
