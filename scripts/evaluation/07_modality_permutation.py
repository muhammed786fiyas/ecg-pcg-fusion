"""Modality permutation test: does a fusion model actually use each modality?

For each fold's TEST records, one modality's input is swapped for the same
modality from a DIFFERENT test record, the other is left intact, and the
record-level AUC is recomputed. If shuffling the PCG barely moves the AUC, the
model is effectively ECG-only however it is wired; a large drop means it relies
on the PCG. This is permutation importance at the record level: the swap breaks
the link between a record's PCG and both its label and its own ECG.

The two branches are independent until cross-attention, so each segment's
branch feature maps are computed once and re-paired for every permutation - the
expensive backbone runs once per fold, not once per shuffle.

Nothing is fitted, so there is nothing to leak: the checkpoints are the trained
CV models and only test records are read. The intact condition must reproduce
the saved test predictions, which proves the re-pairing path is the model's own
forward pass.
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

BATCH_SIZE = 32
HEAD_BATCH_SIZE = 256
N_PERMUTATIONS = 5
# Local CPU fp32 against Kaggle GPU mixed precision: small drift is expected,
# a large one would mean the re-pairing path is not the model's forward pass.
MAX_TEST_PROB_DRIFT = 0.05
CONDITIONS = ["intact", "pcg_shuffled", "ecg_shuffled"]


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def load_family(family):
    path = os.path.join("scripts", "modeling", family, "01_train.py")
    if not os.path.exists(path):
        raise SystemExit(f"QC FAIL: {path} not found")
    spec = importlib.util.spec_from_file_location("train_" + family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def branch_maps(module, model, manifest_path, scalogram_dir):
    """Each segment's ECG and PCG branch feature maps, before any fusion."""
    dataset = module.ScalogramDataset(manifest_path, scalogram_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ecg_parts = []
    pcg_parts = []
    with torch.no_grad():
        for ecg, pcg, _ in loader:
            ecg_parts.append(model.ecg_branch.feature_maps(ecg))
            pcg_parts.append(model.pcg_branch.feature_maps(pcg))
    frame = dataset.manifest[["segment_variant_id", "record_id", "label"]].reset_index(drop=True)
    return frame, torch.cat(ecg_parts), torch.cat(pcg_parts)


def head(model, ecg_maps, pcg_maps):
    """The model's forward pass from the branch feature maps onwards."""
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(ecg_maps), HEAD_BATCH_SIZE):
            stop = start + HEAD_BATCH_SIZE
            ecg_tokens = model.to_tokens(ecg_maps[start:stop])
            pcg_tokens = model.to_tokens(pcg_maps[start:stop])
            ecg_out, pcg_out = model.cross_attention(ecg_tokens, pcg_tokens)
            fused = torch.cat([ecg_out.mean(dim=1), pcg_out.mean(dim=1)], dim=1)
            probabilities.append(torch.sigmoid(model.classifier(fused).squeeze(1)).numpy())
    return np.concatenate(probabilities)


def derangement(n, rng):
    """A random permutation of range(n) with no fixed points."""
    if n < 2:
        raise SystemExit(f"QC FAIL: cannot derange {n} records")
    while True:
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order


def donor_rows(frame, rng):
    """For every segment, the row of a segment from a DIFFERENT record.

    Records are deranged as a whole, so each record takes all of its swapped
    modality from one other record; segment k takes that record's segment k,
    wrapping round when the donor has fewer segments.
    """
    positions = {}
    for index, record in enumerate(frame["record_id"]):
        positions.setdefault(record, []).append(index)
    records = sorted(positions)
    order = derangement(len(records), rng)
    donors = np.zeros(len(frame), dtype=int)
    for index, record in enumerate(records):
        donor_positions = positions[records[order[index]]]
        for k, row in enumerate(positions[record]):
            donors[row] = donor_positions[k % len(donor_positions)]
    return donors


def aucs(frame, probabilities):
    """(segment AUC, record AUC on the mean segment probability)."""
    scored = frame.assign(prob=probabilities)
    records = scored.groupby("record_id").agg(label=("label", "first"), prob=("prob", "mean"))
    return (
        float(roc_auc_score(scored["label"], scored["prob"])),
        float(roc_auc_score(records["label"], records["prob"])),
    )


def check_against_saved(frame, probabilities, saved_path):
    if not os.path.exists(saved_path):
        raise SystemExit(f"QC FAIL: saved test predictions not found: {saved_path}")
    saved = pd.read_csv(saved_path)[["segment_variant_id", "prob"]]
    merged = frame.assign(prob=probabilities).merge(saved, on="segment_variant_id", suffixes=("", "_saved"))
    if len(merged) != len(frame):
        raise SystemExit(f"QC FAIL: {len(frame)} test rows but only {len(merged)} match {saved_path}")
    drift = float(np.max(np.abs(merged["prob"] - merged["prob_saved"])))
    if drift > MAX_TEST_PROB_DRIFT:
        raise SystemExit(f"QC FAIL: intact probabilities drift by up to {round(drift, 4)} from the saved ones")
    return drift


def main():
    parser = argparse.ArgumentParser(description="record-level modality permutation test")
    parser.add_argument("--family", default="cross_attn_resnet18")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--manifest-root", default=os.path.join("data", "processed", "manifests"))
    parser.add_argument("--scalogram-root", default=os.path.join("data", "processed", "scalograms"))
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--output-dir", default=os.path.join("reports", "modality_permutation"))
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    k = params["evaluation"]["cv_folds"]
    seed = params["global_seed"]
    torch.set_num_threads(os.cpu_count())

    print("=== 07_modality_permutation ===")
    print(f"family={args.family} config={args.scalogram_config} folds={k} permutations={N_PERMUTATIONS}")

    module = load_family(args.family)
    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    rows = []
    for fold in range(k):
        tag = "cv_fold" + str(fold)
        checkpoint = os.path.join(
            args.model_root, args.family, args.scalogram_config, f"{args.scalogram_config}_{tag}_best.pth"
        )
        if not os.path.exists(checkpoint):
            raise SystemExit(f"QC FAIL: checkpoint not found: {checkpoint}")
        model = module.build_model(params)
        if not hasattr(model, "cross_attention"):
            raise SystemExit(f"QC FAIL: {args.family} has no cross-attention head to re-pair")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        frame, ecg_maps, pcg_maps = branch_maps(
            module, model, os.path.join(args.manifest_root, tag, "test.csv"), scalogram_dir
        )
        intact = head(model, ecg_maps, pcg_maps)
        saved_path = os.path.join(args.report_root, args.family, args.scalogram_config, tag, "test_predictions.csv")
        drift = check_against_saved(frame, intact, saved_path)
        segment_auc, patient_auc = aucs(frame, intact)
        rows.append({"fold": tag, "condition": "intact", "permutation": 0,
                     "segment_auc": segment_auc, "patient_auc": patient_auc})

        rng = np.random.default_rng(seed + fold)
        for permutation in range(N_PERMUTATIONS):
            donors = donor_rows(frame, rng)
            segment_auc, patient_auc = aucs(frame, head(model, ecg_maps, pcg_maps[donors]))
            rows.append({"fold": tag, "condition": "pcg_shuffled", "permutation": permutation,
                         "segment_auc": segment_auc, "patient_auc": patient_auc})
            donors = donor_rows(frame, rng)
            segment_auc, patient_auc = aucs(frame, head(model, ecg_maps[donors], pcg_maps))
            rows.append({"fold": tag, "condition": "ecg_shuffled", "permutation": permutation,
                         "segment_auc": segment_auc, "patient_auc": patient_auc})

        fold_frame = pd.DataFrame([row for row in rows if row["fold"] == tag])
        means = fold_frame.groupby("condition")["patient_auc"].mean()
        print(
            f"  {tag}: {len(frame)} segments, drift {round(drift, 5)} | patient AUC intact "
            f"{round(means['intact'], 3)}, PCG shuffled {round(means['pcg_shuffled'], 3)}, "
            f"ECG shuffled {round(means['ecg_shuffled'], 3)}"
        )

    frame = pd.DataFrame(rows)
    out_dir = os.path.join(args.output_dir, args.family)
    os.makedirs(out_dir, exist_ok=True)
    frame.to_csv(os.path.join(out_dir, "per_permutation.csv"), index=False)

    # One number per fold and condition: the mean over permutations.
    per_fold = frame.groupby(["fold", "condition"])[["segment_auc", "patient_auc"]].mean().reset_index()
    per_fold.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    intact = per_fold[per_fold["condition"] == "intact"].set_index("fold")

    table_rows = []
    for condition in CONDITIONS:
        subset = per_fold[per_fold["condition"] == condition].set_index("fold")
        drop = intact["patient_auc"] - subset["patient_auc"]
        table_rows.append(
            {
                "condition": condition,
                "segment AUC": f"{subset['segment_auc'].mean():.3f} ± {subset['segment_auc'].std(ddof=0):.3f}",
                "patient AUC": f"{subset['patient_auc'].mean():.3f} ± {subset['patient_auc'].std(ddof=0):.3f}",
                "patient AUC drop": f"{drop.mean():.3f} ± {drop.std(ddof=0):.3f}",
                "folds with a drop": f"{int((drop > 0).sum())}/{k}",
            }
        )
    table = pd.DataFrame(table_rows)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# Modality permutation test — {args.family}\n\n")
        handle.write(
            "Within each fold's test records, one modality is swapped for the same modality "
            "from a different test record (records deranged as a whole) and the record-level "
            f"AUC recomputed; {N_PERMUTATIONS} permutations per fold, averaged, then mean ± std "
            f"across the {k} folds. A large drop means the model relies on that modality. "
            "Nothing is fitted; the intact row reproduces the saved test predictions.\n\n"
        )
        handle.write(table.to_markdown(index=False))
        handle.write("\n")

    print("")
    print(table.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
