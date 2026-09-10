"""Package scalograms, manifests, params and training scripts into a private Kaggle Dataset.

Creates the dataset the first time and versions it thereafter.

Why everything goes up as a single payload.zip: `kaggle datasets create` defaults
to --dir-mode skip, which silently SKIPS subdirectories rather than uploading
them. Uploading one archive and letting the kernel unpack it means the directory
structure survives regardless of how Kaggle handles the archive, and the kernel
entry script handles both the extracted and the still-zipped layout.
"""

import argparse
import json
import os
import shutil
import subprocess
import zipfile

PAYLOAD_NAME = "payload.zip"
METADATA_NAME = "dataset-metadata.json"

TRAINING_FAMILIES = [
    "ecg_only",
    "pcg_only",
    "dual_cnn",
    "warm_start_fusion",
    "cbam_fusion",
    "cross_attn_fusion",
    "cross_attn_resnet18",
]


def run_kaggle(command_args):
    print("running: kaggle " + " ".join(command_args))
    result = subprocess.run(
        ["kaggle", *command_args], capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print("stderr: " + result.stderr.strip())
    return result


def add_tree(archive, source_dir, arc_prefix):
    """Add a directory to the archive under arc_prefix, preserving structure."""
    n_files = 0
    for root, _, files in os.walk(source_dir):
        for name in files:
            full = os.path.join(root, name)
            relative = os.path.relpath(full, source_dir)
            archive.write(full, os.path.join(arc_prefix, relative))
            n_files = n_files + 1
    return n_files


def build_payload(staging_dir, scalogram_configs, manifest_dir, params_path):
    os.makedirs(staging_dir, exist_ok=True)
    payload_path = os.path.join(staging_dir, PAYLOAD_NAME)

    # ZIP_STORED, not ZIP_DEFLATED: the memmaps are already dense uint8 and
    # compress poorly, so deflating them costs minutes of CPU for a few percent.
    archive = zipfile.ZipFile(payload_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)

    total = 0
    for config_dir in scalogram_configs:
        config_name = os.path.basename(config_dir.rstrip("/\\"))
        count = add_tree(archive, config_dir, os.path.join("scalograms", config_name))
        print(f"added scalogram config {config_name}: {count} files")
        total = total + count

    count = add_tree(archive, manifest_dir, "manifests")
    print(f"added manifests: {count} files")
    total = total + count

    for family in TRAINING_FAMILIES:
        script = os.path.join("scripts", "modeling", family, "01_train.py")
        if not os.path.exists(script):
            raise SystemExit(f"QC FAIL: training script missing: {script}")
        archive.write(script, os.path.join("scripts", "modeling", family, "01_train.py"))
        total = total + 1

    archive.write(params_path, "params.yaml")
    total = total + 1

    archive.close()
    size_mb = round(os.path.getsize(payload_path) / (1024.0 * 1024.0), 1)
    print(f"payload: {total} files, {size_mb} MB at {payload_path}")
    return payload_path


def build_checkpoint_payload(staging_dir, models_dir, families):
    """Package the unimodal best checkpoints for warm_start_fusion.

    /kaggle/working is empty at the start of every kernel, so a fold's ecg_only
    and pcg_only checkpoints have to travel to Kaggle as their own dataset.
    Only the _best files go - the _last files exist for resume and are three
    times the size.
    """
    os.makedirs(staging_dir, exist_ok=True)
    payload_path = os.path.join(staging_dir, PAYLOAD_NAME)
    archive = zipfile.ZipFile(payload_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    total = 0
    for family in families:
        family_dir = os.path.join(models_dir, family)
        if not os.path.isdir(family_dir):
            raise SystemExit(f"QC FAIL: {family_dir} not found - train {family} first")
        for root, _, files in os.walk(family_dir):
            for name in files:
                if not name.endswith("_best.pth"):
                    continue
                full = os.path.join(root, name)
                relative = os.path.relpath(full, models_dir)
                archive.write(full, relative)
                total = total + 1
    archive.close()

    if total == 0:
        raise SystemExit("QC FAIL: no _best.pth checkpoints found to package")
    size_mb = round(os.path.getsize(payload_path) / (1024.0 * 1024.0), 1)
    print(f"checkpoint payload: {total} files, {size_mb} MB")
    return payload_path


def write_metadata(staging_dir, username, slug, title):
    metadata = {
        "title": title,
        "id": username + "/" + slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    path = os.path.join(staging_dir, METADATA_NAME)
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"wrote {path}")
    return path


def dataset_exists(username, slug):
    result = run_kaggle(["datasets", "list", "--user", username, "--search", slug])
    return (username + "/" + slug) in result.stdout


def main():
    parser = argparse.ArgumentParser(description="sync scalograms + manifests to a private Kaggle Dataset")
    parser.add_argument("--scalogram-config", action="append", default=[],
                        help="path to a scalogram config dir; repeatable")
    parser.add_argument("--manifest-dir", default="data/processed/manifests")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--staging-dir", default=".kaggle_staging")
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--slug", default=os.environ.get("KAGGLE_DATASET_SLUG", "ecg-pcg-fusion-scalograms"))
    parser.add_argument("--title", default="ECG-PCG fusion scalograms and manifests")
    parser.add_argument("--version-notes", default="update")
    parser.add_argument("--checkpoints-only", action="store_true",
                        help="package the unimodal _best checkpoints instead of scalograms")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--checkpoint-families", default="ecg_only,pcg_only")
    parser.add_argument("--dry-run", action="store_true", help="build the payload but do not upload")
    args = parser.parse_args()

    print("=== 01_sync_dataset ===")
    if args.username == "":
        raise SystemExit("QC FAIL: no Kaggle username. Set KAGGLE_USERNAME in .env or pass --username")

    if os.path.isdir(args.staging_dir):
        shutil.rmtree(args.staging_dir)

    if args.checkpoints_only:
        families = [name.strip() for name in args.checkpoint_families.split(",")]
        print(f"user={args.username} slug={args.slug} checkpoint families={families}")
        build_checkpoint_payload(args.staging_dir, args.models_dir, families)
    else:
        configs = args.scalogram_config
        if len(configs) == 0:
            configs = ["data/processed/scalograms/default"]
        for config_dir in configs:
            if not os.path.isdir(config_dir):
                raise SystemExit(f"QC FAIL: scalogram config dir not found: {config_dir}")
        print(f"user={args.username} slug={args.slug} configs={configs}")
        build_payload(args.staging_dir, configs, args.manifest_dir, args.params)
    write_metadata(args.staging_dir, args.username, args.slug, args.title)

    if args.dry_run:
        print("dry run: payload built, nothing uploaded")
        return

    if dataset_exists(args.username, args.slug):
        print("dataset exists, pushing a new version")
        result = run_kaggle(
            ["datasets", "version", "-p", args.staging_dir, "-m", args.version_notes, "--dir-mode", "skip"]
        )
    else:
        print("dataset does not exist, creating it (private)")
        result = run_kaggle(["datasets", "create", "-p", args.staging_dir, "--dir-mode", "skip"])

    if result.returncode != 0:
        raise SystemExit(f"QC FAIL: kaggle upload failed with code {result.returncode}")
    print("upload finished")


if __name__ == "__main__":
    main()
