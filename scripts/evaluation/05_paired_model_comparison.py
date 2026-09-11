"""Paired per-fold comparison between model families.

Every family is trained on the SAME folds, from the same data, with the same
seed. That makes the fold a matched unit, and a paired comparison is both the
correct analysis and a much more powerful one than eyeballing two mean +/- std
figures whose error bars overlap.

Why this matters here specifically: fold-to-fold variation on this dataset is
large (patient AUC spans roughly 0.73 to 0.97 for a single family, on 81 test
records per fold). Two families can differ by less than that spread while one
of them beats the other on nearly every individual fold - or, as it turns out,
while one of them does NOT. Comparing marginal means would hide both cases.

Reports, per pair:
  - the per-fold difference
  - how many folds each family wins
  - the mean paired difference with its standard deviation
  - a paired t-test and a Wilcoxon signed-rank test

With k = 5 folds these tests have very little power, so they are reported as a
guard against over-claiming a difference, not as evidence for one. A p-value
above 0.05 here means "this experiment cannot distinguish them", not "they are
the same".
"""

import argparse
import itertools
import os

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

DEFAULT_FAMILIES = [
    "ecg_only",
    "pcg_only",
    "dual_cnn",
    "warm_start_fusion",
    "cbam_fusion",
    "cross_attn_fusion",
    "cross_attn_resnet18",
    "resnet18_ecg_only",
    "resnet18_pcg_only",
    "resnet18_pcg_only_pcgpre",
    "cross_attn_resnet18_pcgpre",
]
MIN_FOLDS_FOR_TEST = 3


def fetch_fold_runs(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"QC FAIL: experiment '{experiment_name}' not found at {tracking_uri}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
    if "params.fold" not in runs.columns:
        raise SystemExit("QC FAIL: no per-fold runs in the experiment")
    return runs[runs["params.fold"].notna()].copy()


def scores_by_fold(runs, family, metric, config):
    """Metric keyed by fold for one family, restricted to one scalogram config."""
    subset = runs[runs["params.family_label"] == family]
    if "params.scalogram_config" in subset.columns:
        subset = subset[subset["params.scalogram_config"] == config]
    column = "metrics." + metric
    if column not in subset.columns:
        return {}
    return dict(zip(subset["params.fold"], subset[column].astype(float), strict=True))


def compare(left_scores, right_scores):
    """Paired comparison over the folds both families actually have."""
    shared = sorted(set(left_scores.keys()) & set(right_scores.keys()))
    if len(shared) == 0:
        return None

    left = np.array([left_scores[f] for f in shared])
    right = np.array([right_scores[f] for f in shared])
    differences = left - right

    result = {
        "folds": len(shared),
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "left_wins": int((differences > 0).sum()),
        "right_wins": int((differences < 0).sum()),
        "t_statistic": float("nan"),
        "t_p_value": float("nan"),
        "wilcoxon_p_value": float("nan"),
    }

    if len(shared) >= MIN_FOLDS_FOR_TEST and np.any(differences != 0):
        t_statistic, t_p = stats.ttest_rel(left, right)
        result["t_statistic"] = float(t_statistic)
        result["t_p_value"] = float(t_p)
        try:
            result["wilcoxon_p_value"] = float(stats.wilcoxon(left, right).pvalue)
        except ValueError:
            # Wilcoxon refuses degenerate input, e.g. all differences equal.
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description="paired per-fold comparison between families")
    parser.add_argument("--metric", default="patient_auc")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--baseline", default="", help="compare every family against this one only")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = args.tracking_uri if args.tracking_uri else os.environ.get(
        "MLFLOW_TRACKING_URI", "file:./models/mlflow_tracking")
    experiment_name = args.experiment if args.experiment else os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion")

    print("=== 05_paired_model_comparison ===")
    print(f"metric={args.metric} config={args.scalogram_config}")

    runs = fetch_fold_runs(tracking_uri, experiment_name)
    available = []
    scores = {}
    for family in DEFAULT_FAMILIES:
        family_scores = scores_by_fold(runs, family, args.metric, args.scalogram_config)
        if len(family_scores) > 0:
            available.append(family)
            scores[family] = family_scores
    print(f"families with runs: {available}")
    if len(available) < 2:
        raise SystemExit("QC FAIL: need at least two families to compare")

    if args.baseline:
        if args.baseline not in available:
            raise SystemExit(f"QC FAIL: baseline '{args.baseline}' has no runs")
        pairs = [(f, args.baseline) for f in available if f != args.baseline]
    else:
        pairs = list(itertools.combinations(available, 2))

    rows = []
    for left, right in pairs:
        result = compare(scores[left], scores[right])
        if result is None:
            continue
        result["left"] = left
        result["right"] = right
        rows.append(result)
        verdict = "no distinguishable difference"
        if result["t_p_value"] == result["t_p_value"] and result["t_p_value"] < 0.05:
            verdict = "difference detected"
        print(
            f"  {left} vs {right}: mean diff {result['mean_difference']:+.4f} "
            f"(sd {result['std_difference']:.4f}), wins {result['left_wins']}-{result['right_wins']} "
            f"over {result['folds']} folds, paired t p={result['t_p_value']:.3f} -> {verdict}"
        )

    if len(rows) == 0:
        raise SystemExit("QC FAIL: no comparable pairs")

    frame = pd.DataFrame(rows)[
        ["left", "right", "folds", "left_mean", "right_mean", "mean_difference",
         "std_difference", "left_wins", "right_wins", "t_statistic", "t_p_value", "wilcoxon_p_value"]
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"paired_comparison_{args.metric}.csv")
    frame.to_csv(csv_path, index=False)

    md_path = os.path.join(args.output_dir, f"paired_comparison_{args.metric}.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Paired per-fold comparison — {args.metric}\n\n")
        handle.write(
            "Every family is trained on the same folds, from the same data, with the "
            "same seed, so the fold is a matched unit and the paired difference is the "
            "right statistic. Fold-to-fold variation on this dataset is large, and "
            "comparing two marginal means with overlapping error bars would hide both "
            "real differences and the absence of one.\n\n"
            f"**With only {int(frame['folds'].max())} folds these tests have very little "
            "power.** A p-value above 0.05 here means this experiment cannot "
            "distinguish the two families, NOT that they perform identically.\n\n"
        )
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")

    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
