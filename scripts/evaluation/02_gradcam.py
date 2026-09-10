"""Contribution 2: Grad-CAM on the trained cross_attn_fusion model.

Hooks the last Conv2d of each branch's CNNBranch.features, computes a class
activation map per modality, and overlays it on a jet render of that segment's
scalogram.

Grad-CAM is hand-rolled rather than taken from the `grad-cam` package: it is a
forward hook, a backward hook and four lines of arithmetic, and inlining it
keeps this script runnable inside a Kaggle kernel with no extra dependency.

Segments are chosen to cover correctly classified normal, correctly classified
abnormal, and misclassifications - the last of these matter most for the write
up and are the easiest to quietly leave out.

The interpretation itself goes in docs/logs/tasks/4-interpretability.md, and it
is to be written honestly: if the ECG map does not concentrate on the QRS
time-frequency region and the PCG map on the S1/S2 bursts, that is the result.
"""

import argparse
import importlib.util
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UINT8_MAX = 255.0
IMAGE_SIZE = 224
OVERLAY_ALPHA = 0.45
CAM_EPSILON = 1e-8


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def load_training_module(family):
    path = os.path.join("scripts", "modeling", family, "01_train.py")
    if not os.path.exists(path):
        raise SystemExit(f"QC FAIL: {path} not found")
    spec = importlib.util.spec_from_file_location("train_" + family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_conv_layer(branch):
    """The last Conv2d inside a CNNBranch's feature stack - where Grad-CAM hooks."""
    convs = [layer for layer in branch.features.modules() if isinstance(layer, torch.nn.Conv2d)]
    if len(convs) == 0:
        raise SystemExit("QC FAIL: no Conv2d found in the branch feature stack")
    return convs[-1]


class GradCam:
    """Forward and backward hooks on one conv layer.

    activations: the layer's output for the current forward pass.
    gradients:   d(target logit) / d(that output).
    The map is ReLU over the channel-weighted sum, weights being the spatially
    averaged gradients - Selvaraju et al., ICCV 2017.
    """

    def __init__(self, layer):
        self.activations = None
        self.gradients = None
        self.forward_handle = layer.register_forward_hook(self.save_activations)
        self.backward_handle = layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, layer_input, layer_output):
        self.activations = layer_output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self):
        if self.activations is None or self.gradients is None:
            raise SystemExit("QC FAIL: Grad-CAM hooks captured nothing - did backward() run?")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam[0].cpu().numpy()
        cam = cam - cam.min()
        peak = cam.max()
        if peak > CAM_EPSILON:
            cam = cam / peak
        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def upsample_cam(cam, size):
    """Nearest-free bilinear upsample of the small CAM to the image grid."""
    tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def pick_segments(predictions, threshold, n_wanted):
    """A spread over correct-normal, correct-abnormal and misclassified.

    Misclassifications are taken deliberately rather than by luck: they are the
    most informative panels and the easiest to omit by accident.
    """
    frame = predictions.copy()
    frame["prediction"] = (frame["prob"] >= threshold).astype(int)
    frame["correct"] = frame["prediction"] == frame["label"]

    groups = {
        "correct_normal": frame[(frame["correct"]) & (frame["label"] == 0)].sort_values("prob"),
        "correct_abnormal": frame[(frame["correct"]) & (frame["label"] == 1)].sort_values("prob", ascending=False),
        "false_positive": frame[(~frame["correct"]) & (frame["label"] == 0)].sort_values("prob", ascending=False),
        "false_negative": frame[(~frame["correct"]) & (frame["label"] == 1)].sort_values("prob"),
    }

    per_group = max(1, n_wanted // len(groups))
    chosen = []
    for name, group in groups.items():
        taken = group.head(per_group)
        for row in taken.itertuples(index=False):
            chosen.append({"category": name, "row": row})
        print(f"  {name}: {len(group)} available, took {len(taken)}")
    return chosen[:n_wanted]


def render_panel(ecg_image, pcg_image, ecg_cam, pcg_cam, title, out_path):
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))

    axes[0][0].imshow(ecg_image, cmap="jet", aspect="auto", origin="lower")
    axes[0][0].set_title("ECG scalogram")
    axes[0][1].imshow(ecg_image, cmap="gray", aspect="auto", origin="lower")
    axes[0][1].imshow(ecg_cam, cmap="jet", alpha=OVERLAY_ALPHA, aspect="auto", origin="lower")
    axes[0][1].set_title("ECG Grad-CAM")

    axes[1][0].imshow(pcg_image, cmap="jet", aspect="auto", origin="lower")
    axes[1][0].set_title("PCG scalogram")
    axes[1][1].imshow(pcg_image, cmap="gray", aspect="auto", origin="lower")
    axes[1][1].imshow(pcg_cam, cmap="jet", alpha=OVERLAY_ALPHA, aspect="auto", origin="lower")
    axes[1][1].set_title("PCG Grad-CAM")

    for row in axes:
        for axis in row:
            axis.set_xlabel("time (samples, 3 s window)")
            axis.set_ylabel("scale (low freq at top)")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def cam_mass_profile(cam):
    """Where the map puts its mass, as fractions along each axis.

    Reported so the physiological question can be answered with numbers rather
    than by eyeballing: the frequency profile says which scale bands the model
    attends to, the time profile whether it locks onto discrete events.
    """
    frequency = cam.mean(axis=1)
    time = cam.mean(axis=0)
    thirds = len(frequency) // 3
    return {
        "freq_low_third": float(frequency[:thirds].mean()),
        "freq_mid_third": float(frequency[thirds : 2 * thirds].mean()),
        "freq_high_third": float(frequency[2 * thirds :].mean()),
        "time_peak_position": float(np.argmax(time)) / float(len(time)),
        "time_concentration": float(time.max() / (time.mean() + CAM_EPSILON)),
    }


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM on the headline fusion model")
    parser.add_argument("--family", default="cross_attn_fusion")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--fold-tag", default="cv_fold0")
    parser.add_argument("--scalogram-root", default="data/processed/scalograms")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--output-dir", default="reports/gradcam")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    threshold = params["evaluation"]["decision_threshold"]
    n_segments = params["evaluation"]["gradcam_n_segments"]

    print("=== 02_gradcam ===")
    print(f"family={args.family} config={args.scalogram_config} fold={args.fold_tag} n_segments={n_segments}")

    module = load_training_module(args.family)
    model = module.build_model(params)

    checkpoint = os.path.join(
        args.model_root, args.family, args.scalogram_config,
        f"{args.scalogram_config}_{args.fold_tag}_best.pth",
    )
    if not os.path.exists(checkpoint):
        raise SystemExit(f"QC FAIL: checkpoint not found: {checkpoint}. Train {args.family} first.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {checkpoint} (val_auc={round(float(state.get('val_auc', float('nan'))), 4)})")

    predictions_path = os.path.join(
        args.report_root, args.family, args.scalogram_config, args.fold_tag, "test_predictions.csv"
    )
    if not os.path.exists(predictions_path):
        raise SystemExit(f"QC FAIL: {predictions_path} not found")
    predictions = pd.read_csv(predictions_path)
    print(f"test predictions: {len(predictions)} rows")

    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    row_index = pd.read_csv(os.path.join(scalogram_dir, "row_index.csv"))
    row_of = dict(zip(row_index["segment_variant_id"], row_index["row_index"], strict=True))
    ecg_memmap = np.load(os.path.join(scalogram_dir, "ecg.uint8.npy"), mmap_mode="r")
    pcg_memmap = np.load(os.path.join(scalogram_dir, "pcg.uint8.npy"), mmap_mode="r")

    chosen = pick_segments(predictions, threshold, n_segments)
    print(f"selected {len(chosen)} segments")

    os.makedirs(args.output_dir, exist_ok=True)
    ecg_cam_hook = GradCam(last_conv_layer(model.ecg_branch))
    pcg_cam_hook = GradCam(last_conv_layer(model.pcg_branch))

    records = []
    for entry in chosen:
        row = entry["row"]
        variant_id = row.segment_variant_id
        memmap_row = int(row_of[variant_id])

        ecg_image = np.asarray(ecg_memmap[memmap_row], dtype=np.float32) / UINT8_MAX
        pcg_image = np.asarray(pcg_memmap[memmap_row], dtype=np.float32) / UINT8_MAX
        ecg_tensor = torch.from_numpy(ecg_image).unsqueeze(0).unsqueeze(0)
        pcg_tensor = torch.from_numpy(pcg_image).unsqueeze(0).unsqueeze(0)

        model.zero_grad(set_to_none=True)
        logit = model(ecg_tensor, pcg_tensor)
        logit.backward()

        ecg_cam = upsample_cam(ecg_cam_hook.compute(), IMAGE_SIZE)
        pcg_cam = upsample_cam(pcg_cam_hook.compute(), IMAGE_SIZE)

        probability = float(torch.sigmoid(logit.detach())[0])
        label_name = "abnormal" if int(row.label) == 1 else "normal"
        title = (
            f"{variant_id} | record {row.record_id} | true={label_name} "
            f"| p(abnormal)={round(probability, 3)} | {entry['category']}"
        )
        out_path = os.path.join(args.output_dir, entry["category"] + "_" + variant_id + ".png")
        render_panel(ecg_image, pcg_image, ecg_cam, pcg_cam, title, out_path)

        record = {
            "segment_variant_id": variant_id,
            "record_id": row.record_id,
            "label": int(row.label),
            "prob": probability,
            "category": entry["category"],
            "figure": os.path.basename(out_path),
        }
        for name, value in cam_mass_profile(ecg_cam).items():
            record["ecg_" + name] = value
        for name, value in cam_mass_profile(pcg_cam).items():
            record["pcg_" + name] = value
        records.append(record)
        print(f"  wrote {os.path.basename(out_path)} p={round(probability, 3)}")

    ecg_cam_hook.close()
    pcg_cam_hook.close()

    frame = pd.DataFrame(records)
    summary_path = os.path.join(args.output_dir, "gradcam_summary.csv")
    frame.to_csv(summary_path, index=False)

    print("")
    print("Grad-CAM mass by frequency third (mean over the selected segments):")
    for modality in ["ecg", "pcg"]:
        low = round(float(frame[modality + "_freq_low_third"].mean()), 4)
        mid = round(float(frame[modality + "_freq_mid_third"].mean()), 4)
        high = round(float(frame[modality + "_freq_high_third"].mean()), 4)
        concentration = round(float(frame[modality + "_time_concentration"].mean()), 3)
        print(f"  {modality}: low={low} mid={mid} high={high} time_concentration={concentration}")
    print("")
    print("Note: in these images row 0 is the SMALLEST scale, i.e. the HIGHEST frequency.")
    print("Interpret and write up honestly in docs/logs/tasks/4-interpretability.md")

    with open(os.path.join(args.output_dir, "gradcam_summary.json"), "w") as handle:
        json.dump(records, handle, indent=2)
    print(f"wrote {summary_path} and {len(records)} figures")


if __name__ == "__main__":
    main()
