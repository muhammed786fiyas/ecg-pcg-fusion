"""Sweep every fetched kernel output into the local MLflow store, and report gaps.

Two failure modes this exists to close, both of which produce a results table
that looks complete while quietly missing folds:

1. **A fetched run that never merged.** `kaggle kernels output` has been observed
   returning a non-zero exit code while still delivering every file. The old
   queue treated that as a failed fetch and skipped the merge, so a finished
   fold's checkpoints and MLflow run sat on disk, unmerged, and nothing
   complained. Merging is idempotent, so re-running the sweep is always safe.

2. **An expected (family, fold) with no run at all** - a kernel that errored, or
   was never launched because the queue hit its budget. These are listed
   explicitly, because a 5-fold mean computed over 4 folds is not a 5-fold mean
   and the table has no way to tell.

Run this before building any results table.
"""

import argparse
import glob
import os
import subprocess
import sys

import mlflow
from dotenv import load_dotenv

PYTHON = sys.executable
EXPECTED_FAMILIES = [
    "ecg_only",
    "pcg_only",
    "dual_cnn",
    "warm_start_fusion",
    "cbam_fusion",
    "cross_attn_fusion",
    "cross_attn_resnet18",
]


def local_run_ids(tracking_dir, experiment_id):
    path = os.path.join(tracking_dir, experiment_id)
    if not os.path.isdir(path):
        return set()
    return {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}


def unmerged_outputs(kernel_dir, known_ids):
    """Kernel output directories holding MLflow runs absent from the local store."""
    pending = {}
    pattern = os.path.join(kernel_dir, "*", "output", "models", "mlflow_tracking", "*", "*")
    for path in sorted(glob.glob(pattern)):
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "meta.yaml")):
            continue
        if os.path.basename(path) in known_ids:
            continue
        output_root = path.split(os.sep + "models" + os.sep)[0]
        pending.setdefault(output_root, 0)
        pending[output_root] = pending[output_root] + 1
    return pending


def merge(output_root):
    result = subprocess.run(
        [PYTHON, "scripts/remote/04_merge_results.py", "--kernel-output", output_root],
        capture_output=True, text=True,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()[-300:]


def report_coverage(tracking_uri, experiment_name, expected_folds, config):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("no experiment yet")
        return []
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
    if "params.fold" not in runs.columns:
        print("no per-fold runs yet")
        return []
    folds = runs[runs["params.fold"].notna()]
    if "params.scalogram_config" in folds.columns:
        folds = folds[folds["params.scalogram_config"] == config]

    print("")
    print(f"coverage for scalogram config '{config}':")
    gaps = []
    for family in EXPECTED_FAMILIES:
        have = set(folds[folds["params.family_label"] == family]["params.fold"]) if "params.family_label" in folds.columns else set()
        want = {"cv_fold" + str(i) for i in range(expected_folds)}
        missing = sorted(want - have)
        status = "complete" if len(missing) == 0 else "MISSING " + ",".join(missing)
        print(f"  {family:22s} {len(have)}/{expected_folds}  {status}")
        if len(missing) > 0:
            gaps.append((family, missing))
    return gaps


def main():
    parser = argparse.ArgumentParser(description="merge stragglers and report missing folds")
    parser.add_argument("--kernel-dir", default=".kaggle_kernels")
    parser.add_argument("--tracking-dir", default=os.path.join("models", "mlflow_tracking"))
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--scalogram-config", default="default")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = "file:./" + args.tracking_dir.replace("\\", "/")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion")

    print("=== 06_reconcile ===")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    known = local_run_ids(args.tracking_dir, experiment.experiment_id) if experiment else set()
    print(f"runs already in the local store: {len(known)}")

    pending = unmerged_outputs(args.kernel_dir, known)
    print(f"kernel outputs with unmerged runs: {len(pending)}")
    merged_ok = 0
    for output_root, count in pending.items():
        ok, tail = merge(output_root)
        label = os.path.basename(os.path.dirname(output_root))
        if ok:
            merged_ok = merged_ok + 1
            print(f"  merged {count} run(s) from {label}")
        else:
            print(f"  FAILED to merge {label}: {tail}")

    gaps = report_coverage(tracking_uri, experiment_name, args.expected_folds, args.scalogram_config)

    print("")
    if len(gaps) == 0:
        print("every expected family has all folds present")
    else:
        print("INCOMPLETE FAMILIES - do not report these as k-fold means:")
        for family, missing in gaps:
            print(f"  {family}: missing {missing}")
    print(f"merged {merged_ok} straggler output(s)")


if __name__ == "__main__":
    main()
