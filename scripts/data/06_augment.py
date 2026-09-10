"""Seeded 4x augmentation, one pass over every surviving segment.

Variants, each derived only from its own source segment:
  _orig      byte-identical copy
  _noise     ECG += N(0, ecg_noise_std_frac * std(ecg)); PCG += N(0, pcg_noise_std)
  _scale     ECG *= U(ecg_scale_range); PCG *= U(pcg_scale_range)
  _combined  noise + scaling + a PCG temporal shift of +/- pcg_shift_ms

Named _combined, not _mix. The old log called this "signal mixing", which reads
as cross-sample mixup to a reviewer and invites a leakage objection that does not
apply here - nothing is ever combined across segments.

Two things that matter:

1. The RNG is seeded PER SEGMENT AND VARIANT from a stable hash, never drawn
   from one global stream in a loop. A global stream would make the output
   depend on filesystem iteration order, which destroys reproducibility. This is
   a large part of why the old pipeline had to be rebuilt.

2. Every surviving segment is augmented once, for all records, regardless of
   fold. Each variant depends only on its own source segment, and in k-fold CV
   every record is in the training portion of k-1 folds anyway, so generating
   once and letting the MANIFESTS control usage is both correct and k times
   cheaper than regenerating per fold. Val and test manifests reference _orig
   rows only - that is where the train-only rule is enforced.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import yaml

HASH_HEX_DIGITS = 8


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


def segment_variant_seed(global_seed, segment_id, variant):
    """Deterministic RNG seed from a stable hash of (seed, segment_id, variant).

    The brief's example hashes seed and segment_id. Including the variant too
    means each variant draws from its own independent stream, so adding or
    reordering variants later cannot silently change the values of the existing
    ones - the same reproducibility argument, applied one level down.
    """
    payload = f"{global_seed}:{segment_id}:{variant}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:HASH_HEX_DIGITS]
    return int(digest, 16)


def add_noise(ecg, pcg, rng, ecg_noise_std_frac, pcg_noise_std):
    ecg_std = float(np.std(ecg))
    ecg_out = ecg + rng.normal(0.0, ecg_noise_std_frac * ecg_std, size=len(ecg))
    pcg_out = pcg + rng.normal(0.0, pcg_noise_std, size=len(pcg))
    return ecg_out.astype(np.float32), pcg_out.astype(np.float32)


def apply_scaling(ecg, pcg, rng, ecg_scale_range, pcg_scale_range):
    ecg_factor = rng.uniform(ecg_scale_range[0], ecg_scale_range[1])
    pcg_factor = rng.uniform(pcg_scale_range[0], pcg_scale_range[1])
    return (ecg * ecg_factor).astype(np.float32), (pcg * pcg_factor).astype(np.float32)


def shift_signal(signal, shift_samples):
    """Translate a signal in time, padding with the edge value.

    Edge padding rather than np.roll: rolling wraps the tail of the window onto
    the front, splicing together two moments that were never adjacent. For a
    misalignment augmentation that is an artefact, not a plausible signal.
    """
    if shift_samples == 0:
        return np.array(signal, dtype=np.float32, copy=True)
    result = np.empty_like(signal, dtype=np.float32)
    if shift_samples > 0:
        result[:shift_samples] = signal[0]
        result[shift_samples:] = signal[: len(signal) - shift_samples]
    else:
        cut = -shift_samples
        result[: len(signal) - cut] = signal[cut:]
        result[len(signal) - cut :] = signal[-1]
    return result


def make_variant(ecg, pcg, fs, variant, rng, aug_params):
    if variant == "orig":
        return np.array(ecg, dtype=np.float32, copy=True), np.array(pcg, dtype=np.float32, copy=True)

    if variant == "noise":
        return add_noise(ecg, pcg, rng, aug_params["ecg_noise_std_frac"], aug_params["pcg_noise_std"])

    if variant == "scale":
        return apply_scaling(ecg, pcg, rng, aug_params["ecg_scale_range"], aug_params["pcg_scale_range"])

    if variant == "combined":
        ecg_out, pcg_out = add_noise(
            ecg, pcg, rng, aug_params["ecg_noise_std_frac"], aug_params["pcg_noise_std"]
        )
        ecg_out, pcg_out = apply_scaling(
            ecg_out, pcg_out, rng, aug_params["ecg_scale_range"], aug_params["pcg_scale_range"]
        )
        max_shift = int(round(aug_params["pcg_shift_ms"] * fs / 1000.0))
        shift_samples = int(rng.integers(-max_shift, max_shift + 1))
        pcg_out = shift_signal(pcg_out, shift_samples)
        return ecg_out, pcg_out

    raise SystemExit(f"QC FAIL: unknown augmentation variant {variant}")


def main():
    parser = argparse.ArgumentParser(description="seeded per-segment augmentation")
    parser.add_argument("--segments-dir", required=True)
    parser.add_argument("--index-in", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    global_seed = params["global_seed"]
    aug_params = params["data"]["augment"]
    variants = aug_params["variants"]

    print("=== 06_augment ===")
    print(f"global_seed={global_seed} variants={variants}")
    print(f"ecg_noise_std_frac={aug_params['ecg_noise_std_frac']} pcg_noise_std={aug_params['pcg_noise_std']}")
    print(f"ecg_scale_range={aug_params['ecg_scale_range']} pcg_scale_range={aug_params['pcg_scale_range']} pcg_shift_ms={aug_params['pcg_shift_ms']}")

    index = pd.read_csv(args.index_in)
    print(f"segments in: {len(index)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    rows = []
    for row in index.itertuples(index=False):
        payload = np.load(os.path.join(args.segments_dir, row.segment_id + ".npz"))
        ecg = payload["ecg"]
        pcg = payload["pcg"]
        fs = int(payload["fs"])

        for variant in variants:
            rng = np.random.default_rng(segment_variant_seed(global_seed, row.segment_id, variant))
            ecg_out, pcg_out = make_variant(ecg, pcg, fs, variant, rng, aug_params)

            if not np.isfinite(ecg_out).all() or not np.isfinite(pcg_out).all():
                raise SystemExit(f"QC FAIL: augmentation produced non-finite values for {row.segment_id} {variant}")

            variant_id = row.segment_id + "_" + variant
            np.savez_compressed(
                os.path.join(args.output_dir, variant_id + ".npz"),
                ecg=ecg_out,
                pcg=pcg_out,
                fs=np.int32(fs),
            )
            rows.append(
                {
                    "segment_variant_id": variant_id,
                    "segment_id": row.segment_id,
                    "record_id": row.record_id,
                    "label": int(row.label),
                    "variant": variant,
                }
            )

    augmented = pd.DataFrame(rows)
    augmented.to_csv(args.index_out, index=False)

    print(f"augmented rows written: {len(augmented)} (expected {len(index) * len(variants)})")
    for variant, count in augmented["variant"].value_counts().items():
        print(f"  variant {variant}: {count}")
    print(f"records represented: {augmented['record_id'].nunique()}")
    print(f"row label balance: normal={int((augmented['label'] == 0).sum())} abnormal={int((augmented['label'] == 1).sum())}")

    if len(augmented) != len(index) * len(variants):
        raise SystemExit("QC FAIL: augmented row count does not match segments x variants")

    print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
