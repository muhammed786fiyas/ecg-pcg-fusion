"""Merge a Kaggle kernel's returned output into the local MLflow store and reports.

The MLflow file backend is plain directories, which is exactly why it was chosen
over a tracking server: a remote run merges by copying its run directory into
the local experiment. After this, the results-table builder cannot tell a local
run from a Kaggle one, and does not need to.

Run directories are copied under the LOCAL experiment id, because the experiment
id Kaggle generated is not the one this machine uses.
"""

import argparse
import os
import shutil

import mlflow
from dotenv import load_dotenv

META_NAME = "meta.yaml"


def find_run_dirs(tracking_dir):
    """A run directory is one containing meta.yaml with a run_id, under an experiment."""
    runs = []
    if not os.path.isdir(tracking_dir):
        return runs
    for experiment_id in os.listdir(tracking_dir):
        experiment_path = os.path.join(tracking_dir, experiment_id)
        if not os.path.isdir(experiment_path):
            continue
        for run_id in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_id)
            if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, META_NAME)):
                runs.append((run_id, run_path))
    return runs


def rewrite_meta(meta_path, local_experiment_id, local_run_path):
    """Point a copied run's meta.yaml at this machine's experiment id and path."""
    with open(meta_path) as handle:
        lines = handle.readlines()

    rewritten = []
    for line in lines:
        if line.startswith("experiment_id:"):
            rewritten.append(f"experiment_id: '{local_experiment_id}'\n")
        elif line.startswith("artifact_uri:"):
            rewritten.append("artifact_uri: file:///" + local_run_path.replace("\\", "/") + "/artifacts\n")
        else:
            rewritten.append(line)

    with open(meta_path, "w") as handle:
        handle.writelines(rewritten)


def merge_mlflow(source_tracking_dir, local_tracking_dir, experiment_name):
    mlflow.set_tracking_uri("file:" + local_tracking_dir)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    local_experiment_id = experiment.experiment_id
    print(f"local experiment '{experiment_name}' id={local_experiment_id}")

    runs = find_run_dirs(source_tracking_dir)
    print(f"found {len(runs)} run directories in the returned output")

    merged = 0
    skipped = 0
    for run_id, run_path in runs:
        destination = os.path.join(local_tracking_dir, local_experiment_id, run_id)
        if os.path.exists(destination):
            print(f"  skip {run_id}, already present locally")
            skipped = skipped + 1
            continue
        shutil.copytree(run_path, destination)
        rewrite_meta(os.path.join(destination, META_NAME), local_experiment_id, destination)
        print(f"  merged {run_id}")
        merged = merged + 1

    return merged, skipped


def merge_reports(source_reports, local_reports):
    if not os.path.isdir(source_reports):
        print("no reports/ in the returned output")
        return 0
    copied = 0
    for root, _, files in os.walk(source_reports):
        for name in files:
            source = os.path.join(root, name)
            relative = os.path.relpath(source, source_reports)
            destination = os.path.join(local_reports, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            copied = copied + 1
    print(f"copied {copied} report files into {local_reports}")
    return copied


def merge_checkpoints(source_models, local_models):
    """Bring back the best checkpoints, skipping the resume-only _last files."""
    if not os.path.isdir(source_models):
        return 0
    copied = 0
    for root, _, files in os.walk(source_models):
        for name in files:
            if not name.endswith("_best.pth"):
                continue
            source = os.path.join(root, name)
            relative = os.path.relpath(source, source_models)
            destination = os.path.join(local_models, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            copied = copied + 1
    print(f"copied {copied} best checkpoints into {local_models}")
    return copied


def main():
    parser = argparse.ArgumentParser(description="merge Kaggle kernel output into the local stores")
    parser.add_argument("--kernel-output", required=True, help="folder fetched by 03_run_kernel.py")
    parser.add_argument("--local-tracking-dir", default="models/mlflow_tracking")
    parser.add_argument("--local-models", default="models")
    parser.add_argument("--local-reports", default="reports")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion")

    print("=== 04_merge_results ===")
    if not os.path.isdir(args.kernel_output):
        raise SystemExit(f"QC FAIL: kernel output folder not found: {args.kernel_output}")

    source_tracking = os.path.join(args.kernel_output, "models", "mlflow_tracking")
    local_tracking = os.path.abspath(args.local_tracking_dir)
    os.makedirs(local_tracking, exist_ok=True)

    merged, skipped = merge_mlflow(source_tracking, local_tracking, experiment_name)
    merge_reports(os.path.join(args.kernel_output, "reports"), args.local_reports)
    merge_checkpoints(os.path.join(args.kernel_output, "models"), args.local_models)

    print(f"merged {merged} runs, skipped {skipped} already present")
    if merged == 0 and skipped == 0:
        raise SystemExit("QC FAIL: the returned output contained no MLflow runs at all")


if __name__ == "__main__":
    main()
