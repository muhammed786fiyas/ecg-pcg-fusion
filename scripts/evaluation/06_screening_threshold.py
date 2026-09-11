"""Screening operating point: a decision threshold chosen on validation.

Why. At the default 0.5 the record-level decision (mean of the segment
probabilities) gives cross_attn_resnet18 sensitivity 0.93 and specificity 0.77.
For screening, a missed abnormal record costs more than a false alarm that
leads to a further test, so a lower threshold may be the better operating point.
The tempting alternative - calling a record abnormal if ANY segment is
abnormal - reaches similar sensitivity at far worse specificity (0.51), because
one noisy window out of ~9 is enough to flag a healthy patient.

How. A decision threshold moved off 0.5 is a FITTED statistic, so it is fitted
on each fold's inner-VALIDATION records (split rule 5 in CLAUDE.md): the highest
threshold whose record-level sensitivity reaches the target is chosen, placed
midway to the next lower observed score, and only then applied to that fold's
test records. Choosing it on test would report an operating point nobody could
have picked in advance.

The primary target (params evaluation.screening_target_sensitivity) is the one
the demo deploys. A sweep over evaluation.screening_sweep_targets is reported
beside it, so the cost of asking for more sensitivity is visible - every point
in the sweep is still fitted on validation.

The training runs never saved validation predictions, so they are recomputed
here on CPU from each fold's best checkpoint. Test predictions are recomputed
through the same path and checked against the ones the Kaggle GPU runs saved,
so both sides of the threshold come from one inference path. The record-level
predictions are cached in predictions.csv; --reuse-predictions skips inference.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 32
# Local CPU fp32 against Kaggle GPU mixed precision: small drift is expected,
# a large one would mean the wrong checkpoint or a preprocessing mismatch.
MAX_TEST_PROB_DRIFT = 0.05
PREDICTIONS_FILE = "predictions.csv"


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


def predict(module, model, manifest_path, scalogram_dir):
    """Segment probabilities for every row a manifest names."""
    dataset = module.ScalogramDataset(manifest_path, scalogram_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batches = []
    with torch.no_grad():
        for ecg, pcg, _ in loader:
            batches.append(torch.sigmoid(model(ecg, pcg)).numpy())
    frame = dataset.manifest[["segment_variant_id", "record_id", "label"]].copy()
    frame["prob"] = np.concatenate(batches)
    return frame


def to_records(frame):
    """Mean segment probability per record - the aggregation used throughout."""
    return (
        frame.groupby("record_id")
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .reset_index()
    )


def rates(records, threshold):
    abnormal = records[records["label"] == 1]["prob"]
    normal = records[records["label"] == 0]["prob"]
    sensitivity = float((abnormal >= threshold).mean())
    specificity = float((normal < threshold).mean())
    predictions = (records["prob"] >= threshold).astype(int)
    accuracy = float((predictions == records["label"]).mean())
    return sensitivity, specificity, accuracy


def choose_threshold(records, target):
    """Highest threshold whose sensitivity reaches the target, placed midway to
    the next lower observed score so it does not sit exactly on a record."""
    scores = sorted(records["prob"].unique(), reverse=True)
    for position, score in enumerate(scores):
        if rates(records, score)[0] >= target:
            if position + 1 < len(scores):
                return float((score + scores[position + 1]) / 2.0)
            return float(score)
    return float(scores[-1])


def check_against_saved(test_segments, saved_path):
    """Recomputed test probabilities must match the ones the GPU runs saved."""
    if not os.path.exists(saved_path):
        raise SystemExit(f"QC FAIL: saved test predictions not found: {saved_path}")
    saved = pd.read_csv(saved_path)[["segment_variant_id", "prob"]]
    merged = test_segments.merge(saved, on="segment_variant_id", suffixes=("", "_saved"))
    if len(merged) != len(test_segments):
        raise SystemExit(
            f"QC FAIL: {len(test_segments)} recomputed test rows but only {len(merged)} match {saved_path}"
        )
    drift = float(np.max(np.abs(merged["prob"] - merged["prob_saved"])))
    if drift > MAX_TEST_PROB_DRIFT:
        raise SystemExit(
            f"QC FAIL: recomputed test probabilities drift by up to {round(drift, 4)} from the saved ones"
        )
    return drift


def compute_predictions(args, params, k):
    """Record-level val and test predictions for every fold, one row per record."""
    module = load_family(args.family)
    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    frames = []
    for fold in range(k):
        tag = "cv_fold" + str(fold)
        checkpoint = os.path.join(
            args.model_root, args.family, args.scalogram_config, f"{args.scalogram_config}_{tag}_best.pth"
        )
        if not os.path.exists(checkpoint):
            raise SystemExit(f"QC FAIL: checkpoint not found: {checkpoint}")

        model = module.build_model(params)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        val = to_records(predict(module, model, os.path.join(args.manifest_root, tag, "val.csv"), scalogram_dir))
        test_segments = predict(module, model, os.path.join(args.manifest_root, tag, "test.csv"), scalogram_dir)
        saved_path = os.path.join(args.report_root, args.family, args.scalogram_config, tag, "test_predictions.csv")
        drift = check_against_saved(test_segments, saved_path)
        test = to_records(test_segments)
        print(f"  {tag}: {len(val)} val and {len(test)} test records, max test drift vs saved {round(drift, 5)}")

        val["fold"] = fold
        val["split"] = "val"
        test["fold"] = fold
        test["split"] = "test"
        frames.append(val)
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def evaluate_target(predictions, target, k):
    """Fit the threshold on each fold's validation records, apply it to test."""
    rows = []
    for fold in range(k):
        val = predictions[(predictions["fold"] == fold) & (predictions["split"] == "val")]
        test = predictions[(predictions["fold"] == fold) & (predictions["split"] == "test")]
        if val["label"].nunique() < 2:
            raise SystemExit(f"QC FAIL: fold {fold} inner-validation has only one class")

        threshold = choose_threshold(val, target)
        val_sensitivity, val_specificity, _ = rates(val, threshold)
        sensitivity, specificity, accuracy = rates(test, threshold)
        sensitivity_05, specificity_05, accuracy_05 = rates(test, DEFAULT_THRESHOLD)
        rows.append(
            {
                "fold": "cv_fold" + str(fold),
                "threshold": threshold,
                "val_records": len(val),
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "test_records": len(test),
                "test_sensitivity": sensitivity,
                "test_specificity": specificity,
                "test_accuracy": accuracy,
                "test_sensitivity_at_0.5": sensitivity_05,
                "test_specificity_at_0.5": specificity_05,
                "test_accuracy_at_0.5": accuracy_05,
            }
        )
    return pd.DataFrame(rows)


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def pm(values):
    mean, std = mean_std(values)
    return f"{mean:.3f} ± {std:.3f}"


def main():
    parser = argparse.ArgumentParser(description="validation-chosen screening threshold")
    parser.add_argument("--family", default="cross_attn_resnet18")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--manifest-root", default=os.path.join("data", "processed", "manifests"))
    parser.add_argument("--scalogram-root", default=os.path.join("data", "processed", "scalograms"))
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--output-dir", default=os.path.join("reports", "screening_threshold"))
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="load the cached predictions.csv instead of running inference")
    args = parser.parse_args()

    params = load_params(args.params)
    target = float(params["evaluation"]["screening_target_sensitivity"])
    sweep_targets = [float(t) for t in params["evaluation"]["screening_sweep_targets"]]
    k = params["evaluation"]["cv_folds"]
    torch.set_num_threads(os.cpu_count())

    print("=== 06_screening_threshold ===")
    print(f"family={args.family} config={args.scalogram_config} target sensitivity={target} folds={k}")

    out_dir = os.path.join(args.output_dir, args.family)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, PREDICTIONS_FILE)
    if args.reuse_predictions:
        if not os.path.exists(cache_path):
            raise SystemExit(f"QC FAIL: --reuse-predictions but {cache_path} does not exist")
        predictions = pd.read_csv(cache_path)
        print(f"reused {len(predictions)} cached record predictions from {cache_path}")
    else:
        predictions = compute_predictions(args, params, k)
        predictions.to_csv(cache_path, index=False)

    frame = evaluate_target(predictions, target, k)
    frame.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    for row in frame.to_dict("records"):
        print(
            f"  {row['fold']}: threshold {round(row['threshold'], 3)} (val sens {round(row['val_sensitivity'], 3)}) -> "
            f"test sens {round(row['test_sensitivity'], 3)} spec {round(row['test_specificity'], 3)}   "
            f"[at 0.5: sens {round(row['test_sensitivity_at_0.5'], 3)} spec {round(row['test_specificity_at_0.5'], 3)}]"
        )

    sweep_rows = []
    for sweep_target in sweep_targets:
        result = evaluate_target(predictions, sweep_target, k)
        threshold_mean, threshold_std = mean_std(result["threshold"])
        sens_mean, sens_std = mean_std(result["test_sensitivity"])
        spec_mean, spec_std = mean_std(result["test_specificity"])
        sweep_rows.append(
            {
                "target_sensitivity": sweep_target,
                "threshold_mean": threshold_mean, "threshold_std": threshold_std,
                "threshold_fold0": float(result["threshold"].iloc[0]),
                "sensitivity_mean": sens_mean, "sensitivity_std": sens_std,
                "specificity_mean": spec_mean, "specificity_std": spec_std,
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)

    sens_mean, sens_std = mean_std(frame["test_sensitivity"])
    spec_mean, spec_std = mean_std(frame["test_specificity"])
    acc_mean, acc_std = mean_std(frame["test_accuracy"])
    sens05_mean, sens05_std = mean_std(frame["test_sensitivity_at_0.5"])
    spec05_mean, spec05_std = mean_std(frame["test_specificity_at_0.5"])
    acc05_mean, acc05_std = mean_std(frame["test_accuracy_at_0.5"])
    threshold_mean, threshold_std = mean_std(frame["threshold"])

    thresholds = {}
    for fold in range(k):
        thresholds[str(fold)] = float(frame["threshold"].iloc[fold])

    summary = {
        "family": args.family,
        "scalogram_config": args.scalogram_config,
        "target_sensitivity": target,
        "selection": "highest threshold reaching the target on each fold's inner-validation records",
        "thresholds": thresholds,
        "threshold_mean": threshold_mean,
        "threshold_std": threshold_std,
        "cv_test_screening": {
            "sensitivity_mean": sens_mean, "sensitivity_std": sens_std,
            "specificity_mean": spec_mean, "specificity_std": spec_std,
            "accuracy_mean": acc_mean, "accuracy_std": acc_std,
        },
        "cv_test_at_0.5": {
            "sensitivity_mean": sens05_mean, "sensitivity_std": sens05_std,
            "specificity_mean": spec05_mean, "specificity_std": spec05_std,
            "accuracy_mean": acc05_mean, "accuracy_std": acc05_std,
        },
        "per_fold": frame.to_dict("records"),
        "sweep": sweep_rows,
    }
    with open(os.path.join(out_dir, "thresholds.json"), "w") as handle:
        # default=float: pandas can hand back numpy scalars, which json refuses.
        json.dump(summary, handle, indent=2, default=float)

    table = pd.DataFrame(
        [
            {"operating point": "default, threshold 0.5",
             "sensitivity": pm(frame["test_sensitivity_at_0.5"]),
             "specificity": pm(frame["test_specificity_at_0.5"]),
             "accuracy": pm(frame["test_accuracy_at_0.5"])},
            {"operating point": f"screening, threshold {threshold_mean:.3f} ± {threshold_std:.3f}",
             "sensitivity": pm(frame["test_sensitivity"]),
             "specificity": pm(frame["test_specificity"]),
             "accuracy": pm(frame["test_accuracy"])},
        ]
    )
    sweep_table = pd.DataFrame(
        [
            {"validation target": f"{row['target_sensitivity']:.0%}",
             "threshold": f"{row['threshold_mean']:.3f} ± {row['threshold_std']:.3f}",
             "fold-0 threshold": f"{row['threshold_fold0']:.3f}",
             "test sensitivity": f"{row['sensitivity_mean']:.3f} ± {row['sensitivity_std']:.3f}",
             "test specificity": f"{row['specificity_mean']:.3f} ± {row['specificity_std']:.3f}"}
            for row in sweep_rows
        ]
    )
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# Screening operating point — {args.family}\n\n")
        handle.write(
            f"Record-level decision on the mean segment probability. The screening "
            f"threshold is chosen per fold on **inner-validation** records as the highest "
            f"threshold reaching {target:.0%} sensitivity, then applied to that fold's test "
            f"records. Mean ± std across the {k} test folds.\n\n"
        )
        handle.write(table.to_markdown(index=False))
        handle.write("\n\n## Sweep over the validation target\n\n")
        handle.write(
            "Every row is fitted on validation exactly as above; only the target changes. "
            "The fold-0 threshold is the one the demo would deploy for that target.\n\n"
        )
        handle.write(sweep_table.to_markdown(index=False))
        handle.write("\n")

    print("")
    print(f"test, mean over folds: at 0.5 sens {sens05_mean:.3f} spec {spec05_mean:.3f} | "
          f"screening (t {threshold_mean:.3f}) sens {sens_mean:.3f} spec {spec_mean:.3f}")
    for row in sweep_rows:
        print(f"  sweep target {row['target_sensitivity']}: threshold {round(row['threshold_mean'], 3)} "
              f"(fold 0 {round(row['threshold_fold0'], 3)}) -> sens {round(row['sensitivity_mean'], 3)} "
              f"spec {round(row['specificity_mean'], 3)}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
