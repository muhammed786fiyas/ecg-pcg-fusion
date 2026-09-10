"""Contribution 3: segment-to-patient aggregation.

Groups test predictions by record_id and compares three strategies - majority
vote, mean probability, max probability - reporting patient-level accuracy, F1,
AUC, sensitivity and specificity alongside the segment-level numbers.

Under the CV protocol this aggregates within each fold's test partition and then
reports mean +/- std across folds. Pooling every fold's predictions into one set
before aggregating would be wrong: the folds come from different models, and the
spread across folds is exactly the quantity being reported.

Reads the per-fold test_predictions.csv that the training scripts wrote, so it
does not need to reload a model.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score

STRATEGIES = ["majority_vote", "mean_prob", "max_prob"]


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def classification_metrics(labels, scores, predictions):
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    sensitivity = true_positive / float(true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / float(true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "accuracy": float((predictions == labels).mean()),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": safe_auc(labels, scores),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "n": int(len(labels)),
    }


def aggregate(frame, strategy, threshold):
    """Collapse segment rows to one row per record under the given strategy.

    Returns (labels, scores, predictions). For majority_vote the score used for
    AUC is the fraction of segments voting abnormal, which is the natural
    continuous analogue of the vote and keeps AUC meaningful.
    """
    frame = frame.copy()
    frame["segment_prediction"] = (frame["prob"] >= threshold).astype(int)

    if strategy == "mean_prob":
        grouped = frame.groupby("record_id").agg(label=("label", "first"), score=("prob", "mean"))
        predictions = (grouped["score"] >= threshold).astype(int)
    elif strategy == "max_prob":
        grouped = frame.groupby("record_id").agg(label=("label", "first"), score=("prob", "max"))
        predictions = (grouped["score"] >= threshold).astype(int)
    elif strategy == "majority_vote":
        grouped = frame.groupby("record_id").agg(
            label=("label", "first"), score=("segment_prediction", "mean")
        )
        predictions = (grouped["score"] > 0.5).astype(int)
    else:
        raise SystemExit(f"QC FAIL: unknown aggregation strategy {strategy}")

    return grouped["label"].to_numpy(), grouped["score"].to_numpy(), predictions.to_numpy()


def find_fold_dirs(report_root, family, config):
    base = os.path.join(report_root, family, config)
    if not os.path.isdir(base):
        return []
    found = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, "test_predictions.csv")
        if os.path.exists(path):
            found.append((name, path))
    return found


def summarize(per_fold, level, strategy):
    """Mean +/- std across folds. Never the best fold."""
    rows = [entry for entry in per_fold if entry["level"] == level and entry["strategy"] == strategy]
    if len(rows) == 0:
        return None
    summary = {"level": level, "strategy": strategy, "n_folds": len(rows)}
    for metric in ["accuracy", "f1", "auc", "sensitivity", "specificity"]:
        values = [row[metric] for row in rows]
        summary[metric + "_mean"] = float(np.nanmean(values))
        summary[metric + "_std"] = float(np.nanstd(values))
    return summary


def main():
    parser = argparse.ArgumentParser(description="segment-to-patient aggregation comparison")
    parser.add_argument("--family", action="append", default=[], help="model family; repeatable")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--output-dir", default="reports/patient_aggregation")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    threshold = params["evaluation"]["decision_threshold"]
    families = args.family if len(args.family) else ["cross_attn_fusion", "dual_cnn"]

    print("=== 01_patient_aggregation ===")
    print(f"families={families} config={args.scalogram_config} threshold={threshold}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    all_summaries = []

    for family in families:
        fold_dirs = find_fold_dirs(args.report_root, family, args.scalogram_config)
        if len(fold_dirs) == 0:
            print(f"  {family}: no test_predictions.csv found, skipping")
            continue
        print(f"  {family}: {len(fold_dirs)} fold(s)")

        per_fold = []
        for fold_tag, path in fold_dirs:
            frame = pd.read_csv(path)
            if len(frame) == 0:
                print(f"    {fold_tag}: empty predictions, skipping")
                continue

            segment_predictions = (frame["prob"].to_numpy() >= threshold).astype(int)
            segment = classification_metrics(
                frame["label"].to_numpy(), frame["prob"].to_numpy(), segment_predictions
            )
            segment.update({"family": family, "fold": fold_tag, "level": "segment", "strategy": "none"})
            per_fold.append(segment)
            all_rows.append(segment)

            for strategy in STRATEGIES:
                labels, scores, predictions = aggregate(frame, strategy, threshold)
                metrics = classification_metrics(labels, scores, predictions)
                metrics.update({"family": family, "fold": fold_tag, "level": "patient", "strategy": strategy})
                per_fold.append(metrics)
                all_rows.append(metrics)

            best = max(STRATEGIES, key=lambda s: next(r for r in per_fold if r["fold"] == fold_tag and r["strategy"] == s)["auc"])
            print(f"    {fold_tag}: segment auc={round(segment['auc'], 4)} ({segment['n']} rows) best patient strategy={best}")

        for strategy in ["none", *STRATEGIES]:
            level = "segment" if strategy == "none" else "patient"
            summary = summarize(per_fold, level, strategy)
            if summary is not None:
                summary["family"] = family
                all_summaries.append(summary)

    if len(all_rows) == 0:
        raise SystemExit("QC FAIL: no predictions found for any requested family")

    per_fold_frame = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(all_summaries)
    per_fold_path = os.path.join(args.output_dir, "per_fold_metrics.csv")
    summary_path = os.path.join(args.output_dir, "aggregation_summary.csv")
    per_fold_frame.to_csv(per_fold_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    print("")
    print("summary (mean +/- std across folds):")
    for row in summary_frame.itertuples(index=False):
        print(
            f"  {row.family} {row.level}/{row.strategy}: "
            f"auc={round(row.auc_mean, 4)}+/-{round(row.auc_std, 4)} "
            f"acc={round(row.accuracy_mean, 4)} f1={round(row.f1_mean, 4)} "
            f"sens={round(row.sensitivity_mean, 4)} spec={round(row.specificity_mean, 4)} "
            f"({row.n_folds} folds)"
        )

    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"wrote {per_fold_path} and {summary_path}")


if __name__ == "__main__":
    main()
