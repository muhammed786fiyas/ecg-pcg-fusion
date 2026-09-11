"""Stage (and optionally upload) the public HuggingFace Space.

Assembles a self-contained Space folder:
  app.py              scripts/serving/hf_space_app.py
  requirements.txt    CPU torch + the preprocessing stack
  README.md           Spaces front matter + an honest model card
  model/              one slim cross_attn_resnet18 checkpoint
  examples/           two full records from that model's HELD-OUT test fold

Which weights ship: the five cross-validation checkpoints each record the
inner-validation AUC that early stopping chose them on. The one with the highest
VALIDATION AUC is deployed. Test AUC is never consulted - picking the best test
fold would put an optimistic model in front of the public and call it the
paper's result.

Which examples ship: records from the deployed fold's TEST partition only, the
first normal and the first abnormal by record ID - not chosen by how well the
model does on them. Anything else would demo the model on data it trained on.

The token is read from the HF_TOKEN environment variable or a prior
`hf auth login`; it is never printed or written to disk by this script.
"""

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch

APP_SOURCE = os.path.join("scripts", "serving", "hf_space_app.py")
CHECKPOINT_GLOB = os.path.join("models", "cross_attn_resnet18", "default", "default_cv_fold*_best.pth")
FOLD_ASSIGNMENTS = os.path.join("data", "processed", "manifests", "fold_assignments.csv")
RECORDS_DIR = os.path.join("data", "interim", "records")
GITHUB_URL = "https://github.com/muhammed786fiyas/ecg-pcg-fusion"
GRADIO_VERSION = "6.26.0"

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
numpy<2
scipy
pandas
PyWavelets
neurokit2==0.2.7
wfdb
matplotlib
Pillow
"""

README_TEMPLATE = """---
title: ECG-PCG Cardiac Abnormality Detection
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
pinned: false
---

# ECG-PCG Cardiac Abnormality Detection

Dual-branch **CWT scalogram fusion** of a synchronous ECG and phonocardiogram
(PCG), with a pretrained **ResNet-18** per branch and **bidirectional cross-modal
attention** between them. Upload a recording and get a record-level probability
of abnormality, plus Grad-CAM heatmaps over both scalograms.

> **Research demo - not a medical device.** Do not use it for any clinical decision.

## How it decides

1. R-peaks are detected (NeuroKit2, Pan-Tompkins) and the recording is cut into
   non-overlapping 3 s windows centred on beats - exactly as in training.
2. Each window becomes two CWT scalograms: ECG (`cmor1.5-1.0`, 4-100 Hz) and
   PCG (`morl`, 12.5-232 Hz), reflect-padded to suppress boundary artifacts.
3. The model scores every window; the record-level probability is their mean.

## How good is it - and how that was measured

Evaluated on PhysioNet/CinC 2016 Training-A (405 records with synchronous ECG
and PCG) using **record-level stratified 5-fold cross-validation**: no recording
ever contributes segments to both training and test. Patient-level AUC, mean +/-
std across folds:

| model | patient AUC |
|---|---|
| ECG-only CNN | 0.864 +/- 0.085 |
| PCG-only CNN | 0.652 +/- 0.092 |
| Dual-branch CNN, concatenation | 0.850 +/- 0.085 |
| Dual-branch CNN, cross-attention | 0.851 +/- 0.081 |
| **ResNet-18 + cross-attention (this demo)** | **0.938 +/- 0.039** |

The deployed weights are **one** of the five cross-validation models (fold
{fold}), chosen by its inner-validation AUC ({val_auc:.4f}) and never by test
performance. The two example records come from that model's **held-out test
fold**, picked by record ID *before* looking at the model's output - so they are
not guaranteed to be classified correctly, and the demo shows each example's
ground truth next to the prediction. Its most common error is calling normal
records abnormal (patient-level specificity 0.77 vs sensitivity 0.93).

## Limitations worth knowing

- **One dataset, one site, 405 recordings, 71% abnormal.** Performance on other
  populations, devices or recording conditions is unknown.
- **Interpretability is mixed.** In the study's Grad-CAM analysis the PCG branch
  attends to the S1/S2 heart-sound band, as physiology predicts, but the ECG
  branch attends to the low-frequency (under 10 Hz) envelope of the cardiac
  cycle rather than to the QRS complex. The heatmaps here come from a 7 x 7
  feature map and are coarse by construction.
- The input must be **2000 Hz** with both an ECG and a PCG channel.

## Data and code

Trained on the PhysioNet/CinC Challenge 2016 dataset (Liu et al., *Physiological
Measurement* 37(12), 2016; Goldberger et al., *Circulation* 101(23), 2000),
available from PhysioNet under its stated licence. The example recordings are
derived from that dataset and are redistributed with this attribution.

Code, training pipeline and full results: {github_url}
"""


def select_checkpoint():
    """Highest inner-VALIDATION AUC among the CV checkpoints. Test is never read."""
    candidates = []
    for path in sorted(glob.glob(CHECKPOINT_GLOB)):
        state = torch.load(path, map_location="cpu", weights_only=False)
        fold = int(os.path.basename(path).split("fold")[1].split("_")[0])
        candidates.append((float(state["val_auc"]), fold, path, state["model"]))
        print(f"  fold {fold}: inner-validation AUC {round(float(state['val_auc']), 4)}")
    if len(candidates) == 0:
        raise SystemExit(f"QC FAIL: no checkpoints match {CHECKPOINT_GLOB}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0]


def export_examples(fold, out_dir):
    """First normal and first abnormal record of the deployed fold's TEST partition."""
    folds = pd.read_csv(FOLD_ASSIGNMENTS)
    test = folds[folds["cv_fold"] == fold].sort_values("record_id")
    written = []
    for label, name in [(0, "normal"), (1, "abnormal")]:
        pick = test[test["label"] == label]
        if len(pick) == 0:
            raise SystemExit(f"QC FAIL: fold {fold} test partition has no {name} record")
        record_id = pick["record_id"].iloc[0]
        payload = np.load(os.path.join(RECORDS_DIR, record_id + ".npz"))
        frame = pd.DataFrame({"ecg": payload["ecg"], "pcg": payload["pcg"]})
        path = os.path.join(out_dir, f"{name}_{record_id}.csv")
        frame.to_csv(path, index=False, float_format="%.6f")
        written.append(path)
        print(f"  example {name}: {record_id} ({round(len(frame) / 2000.0, 1)} s)")
    return written


def stage(staging_dir):
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(os.path.join(staging_dir, "model"))
    os.makedirs(os.path.join(staging_dir, "examples"))

    print("selecting weights by inner-validation AUC:")
    val_auc, fold, source, state_dict = select_checkpoint()
    print(f"deploying fold {fold} ({source}), validation AUC {round(val_auc, 4)}")

    # Slim checkpoint: the state dict alone, so the Space can load it with
    # weights_only=True instead of unpickling arbitrary objects.
    model_path = os.path.join(staging_dir, "model", "cross_attn_resnet18.pth")
    torch.save({"model": state_dict}, model_path)
    print(f"  wrote {model_path} ({round(os.path.getsize(model_path) / 1048576.0, 1)} MB)")

    shutil.copy(APP_SOURCE, os.path.join(staging_dir, "app.py"))
    with open(os.path.join(staging_dir, "requirements.txt"), "w", newline="\n") as handle:
        handle.write(REQUIREMENTS)
    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(README_TEMPLATE.format(
            gradio_version=GRADIO_VERSION, fold=fold, val_auc=val_auc, github_url=GITHUB_URL,
        ))

    export_examples(fold, os.path.join(staging_dir, "examples"))
    return fold


def upload(staging_dir, space_id, private):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    who = api.whoami()
    print(f"authenticated to HuggingFace as {who['name']}")
    if "/" not in space_id:
        space_id = who["name"] + "/" + space_id

    try:
        api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio",
                        private=private, exist_ok=True)
    except HfHubHTTPError as err:
        status = getattr(err.response, "status_code", None)
        if status == 402:
            # Observed 2026-09-11: HuggingFace now requires a PRO subscription
            # to host Gradio or Docker Spaces, even on the free cpu-basic
            # hardware. Only static Spaces remain free. Nothing is created.
            raise SystemExit(
                "QC FAIL: HuggingFace returned 402 Payment Required. Gradio Spaces now "
                "need a PRO subscription (https://huggingface.co/pro); only static "
                "Spaces are free. No Space was created. The staged folder is ready to "
                "upload once the account can host it."
            )
        raise
    api.upload_folder(folder_path=staging_dir, repo_id=space_id, repo_type="space",
                      commit_message="Deploy ECG-PCG cross_attn_resnet18 demo")
    print(f"uploaded. Space: https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(description="stage and optionally upload the HF Space")
    parser.add_argument("--staging-dir", default=".hf_space_staging")
    parser.add_argument("--space-id", default="ecg-pcg-cardiac-demo",
                        help="<space-name> or <user>/<space-name>")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print("=== build_hf_space ===")
    stage(args.staging_dir)
    files = sorted(os.listdir(args.staging_dir))
    print(f"staged: {files}")

    if args.upload:
        upload(args.staging_dir, args.space_id, args.private)
    else:
        print("staged only. Re-run with --upload once a HuggingFace token is configured.")


if __name__ == "__main__":
    main()
