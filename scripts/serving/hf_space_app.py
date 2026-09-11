"""Public HuggingFace Spaces demo: ECG + PCG -> abnormality probability + Grad-CAM.

Model: cross_attn_resnet18, the strongest family in the study (patient AUC
0.938 +/- 0.039 over record-level stratified 5-fold cross-validation on
PhysioNet/CinC 2016 Training-A). The deployed weights are ONE of the five
cross-validation models, selected by inner-VALIDATION AUC and never by test
performance.

The input pipeline matches training exactly, because a model fed a different
distribution than it was trained on fails quietly:
  1. R-peaks via NeuroKit2 Pan-Tompkins; non-overlapping 3 s windows centred on
     them (scripts/data/03_segment.py)
  2. reflect-padded CWT -> |coeffs| -> antialiased resize to 224 -> per-image
     uint8 (scripts/features/01_scalogram.py)
  3. one probability per window, aggregated to a record-level probability by
     the mean - the best of the three strategies compared in the study

The decision threshold is read from model/decision.json: the highest threshold
that reached the target sensitivity on the deployed fold's inner-validation
records (scripts/evaluation/06_screening_threshold.py). Without that file it
falls back to 0.5. The text shown to users follows the actual value - it does
not assume the fitted threshold came out below 0.5, because it need not.

This file is staged into the Space as app.py by build_hf_space.py.

Research demo. NOT a medical device.
"""

import json
import os
import shutil
import tempfile

import gradio as gr
import matplotlib
import neurokit2 as nk
import numpy as np
import pywt
import torch
import torch.nn as nn
import torchvision
import wfdb
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "cross_attn_resnet18.pth"))
DECISION_PATH = os.environ.get("DECISION_PATH", os.path.join("model", "decision.json"))
EXAMPLE_DIR = "examples"

FS = 2000
WINDOW_SAMPLES = 6000
IMAGE_SIZE = 224
UINT8_MAX = 255.0
ECG_WAVELET = "cmor1.5-1.0"
ECG_SCALES = np.arange(20, 501)
PCG_WAVELET = "morl"
PCG_SCALES = np.arange(7, 131)
# Must equal PAD_SCALE_FACTOR in scripts/features/01_scalogram.py.
PAD_SCALE_FACTOR = 4
PEAK_METHOD = "pantompkins1985"

DROPOUT = 0.5
N_HEADS = 4
PROJ_DIM = 256
DEFAULT_THRESHOLD = 0.5
CAM_EPSILON = 1e-8
OVERLAY_ALPHA = 0.45
N_FREQ_TICKS = 5

DISCLAIMER = (
    "**Research demo - not a medical device.** Do not use it to make any "
    "clinical decision."
)

STATE = {"model": None, "decision": {"threshold": DEFAULT_THRESHOLD}}


# ---------------------------------------------------------------- model


class BidirectionalCrossAttention(nn.Module):
    """ECG tokens query PCG keys/values and vice versa."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.ecg_from_pcg = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.pcg_from_ecg = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ecg_norm = nn.LayerNorm(dim)
        self.pcg_norm = nn.LayerNorm(dim)

    def forward(self, ecg_tokens, pcg_tokens):
        attended_ecg = self.ecg_from_pcg(ecg_tokens, pcg_tokens, pcg_tokens)[0]
        attended_pcg = self.pcg_from_ecg(pcg_tokens, ecg_tokens, ecg_tokens)[0]
        return self.ecg_norm(ecg_tokens + attended_ecg), self.pcg_norm(pcg_tokens + attended_pcg)


class ResNetBranch(nn.Module):
    """ResNet-18 trunk, 1-channel input replicated to 3, projected to 256."""

    def __init__(self, proj_dim):
        super().__init__()
        # weights=None: every backbone weight lives in the fine-tuned checkpoint,
        # so fetching ImageNet weights at startup would be a wasted download.
        backbone = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.project = nn.Conv2d(512, proj_dim, 1)

    def feature_maps(self, x):
        return self.project(self.stem(x.repeat(1, 3, 1, 1)))


class CrossAttnResNet18(nn.Module):
    def __init__(self, dropout, n_heads, proj_dim):
        super().__init__()
        self.ecg_branch = ResNetBranch(proj_dim)
        self.pcg_branch = ResNetBranch(proj_dim)
        self.cross_attention = BidirectionalCrossAttention(proj_dim, n_heads)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def to_tokens(self, feature_map):
        return feature_map.flatten(2).transpose(1, 2)

    def forward(self, ecg, pcg):
        ecg_tokens = self.to_tokens(self.ecg_branch.feature_maps(ecg))
        pcg_tokens = self.to_tokens(self.pcg_branch.feature_maps(pcg))
        ecg_out, pcg_out = self.cross_attention(ecg_tokens, pcg_tokens)
        fused = torch.cat([ecg_out.mean(dim=1), pcg_out.mean(dim=1)], dim=1)
        return self.classifier(fused).squeeze(1)


def load_model():
    model = CrossAttnResNet18(DROPOUT, N_HEADS, PROJ_DIM)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {MODEL_PATH}")
    return model


def load_decision():
    """The decision threshold and its cross-validated rates, if staged."""
    if not os.path.exists(DECISION_PATH):
        print(f"no {DECISION_PATH}, using threshold {DEFAULT_THRESHOLD}")
        return {"threshold": DEFAULT_THRESHOLD}
    with open(DECISION_PATH) as handle:
        decision = json.load(handle)
    print(f"decision threshold {round(decision['threshold'], 3)} from {DECISION_PATH}")
    return decision


# ---------------------------------------------------------- preprocessing


def resize_to_square(field):
    return np.asarray(Image.fromarray(field, mode="F").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))


def to_unit_image(field):
    """Per-image min/max to uint8 and back, exactly as the training memmap."""
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low <= 0:
        return np.zeros(field.shape, dtype=np.float32)
    quantised = np.clip(((field - low) / (high - low)) * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)
    return quantised.astype(np.float32) / UINT8_MAX


def compute_scalogram(signal, scales, wavelet):
    """Reflect-padded CWT, cropped back to the window - identical to training."""
    pad = int(min(len(signal), PAD_SCALE_FACTOR * int(np.max(scales))))
    padded = signal
    if pad > 0:
        padded = np.pad(signal, pad, mode="reflect")
    coeffs = pywt.cwt(padded, scales, wavelet, sampling_period=1.0 / FS, method="fft")[0]
    magnitude = np.abs(coeffs).astype(np.float32)
    if pad > 0:
        magnitude = magnitude[:, pad : pad + len(signal)]
    return to_unit_image(resize_to_square(magnitude))


def fill_non_finite(signal):
    result = np.array(signal, dtype=np.float32, copy=True)
    bad = ~np.isfinite(result)
    if bad.any() and (~bad).any():
        positions = np.arange(len(result))
        result[bad] = np.interp(positions[bad], positions[~bad], result[~bad])
    return result


def find_windows(ecg):
    """Non-overlapping 3 s windows centred on R-peaks, as in 03_segment.py."""
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=FS, method=PEAK_METHOD)
    except (ValueError, IndexError, ZeroDivisionError):
        return []
    half = WINDOW_SAMPLES // 2
    windows = []
    next_free = 0
    for peak in np.asarray(info["ECG_R_Peaks"], dtype=int):
        start = int(peak) - half
        stop = start + WINDOW_SAMPLES
        if start < next_free or start < 0 or stop > len(ecg):
            continue
        windows.append((start, stop))
        next_free = stop
    return windows


# ---------------------------------------------------------------- input


def as_paths(files):
    if files is None:
        return []
    if not isinstance(files, list):
        files = [files]
    return [getattr(item, "name", item) for item in files]


def read_upload(files):
    """Return (ecg, pcg) at 2000 Hz, or raise ValueError a user can act on."""
    paths = as_paths(files)
    if len(paths) == 0:
        raise ValueError("Upload a two-column CSV (ecg,pcg) or a WFDB .hea + .dat pair.")

    csv_paths = [p for p in paths if p.lower().endswith(".csv")]
    if len(csv_paths) > 0:
        data = np.loadtxt(csv_paths[0], delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("The CSV needs two columns with a header row: ecg,pcg")
        return data[:, 0], data[:, 1]

    headers = [p for p in paths if p.lower().endswith(".hea")]
    if len(headers) == 0:
        raise ValueError("No .csv or .hea file found in the upload.")

    # Gradio may place each upload in its own temporary folder, and wfdb needs
    # the header and signal files side by side.
    work_dir = tempfile.mkdtemp()
    for path in paths:
        shutil.copy(path, os.path.join(work_dir, os.path.basename(path)))
    record_name = os.path.splitext(os.path.basename(headers[0]))[0]
    record = wfdb.rdrecord(os.path.join(work_dir, record_name))

    names = [name.strip().upper() for name in record.sig_name]
    if "ECG" not in names or "PCG" not in names:
        raise ValueError(f"The record needs both ECG and PCG channels; it has {record.sig_name}.")
    if int(record.fs) != FS:
        raise ValueError(f"The model expects {FS} Hz; this record is {int(record.fs)} Hz.")
    ecg = record.p_signal[:, names.index("ECG")].astype(np.float32)
    pcg = record.p_signal[:, names.index("PCG")].astype(np.float32)
    return ecg, pcg


# ------------------------------------------------------------- Grad-CAM


def last_conv(branch):
    convs = [module for module in branch.modules() if isinstance(module, nn.Conv2d)]
    return convs[-1]


def save_activation(store, key):
    def hook(module, layer_input, layer_output):
        store[key] = layer_output.detach()
    return hook


def save_gradient(store, key):
    def hook(module, grad_input, grad_output):
        store[key] = grad_output[0].detach()
    return hook


def grad_cam_pair(model, ecg_tensor, pcg_tensor):
    """Grad-CAM on the last conv of each branch, from one backward pass."""
    captured = {}
    handles = []
    for name, branch in [("ecg", model.ecg_branch), ("pcg", model.pcg_branch)]:
        layer = last_conv(branch)
        handles.append(layer.register_forward_hook(save_activation(captured, name + "_a")))
        handles.append(layer.register_full_backward_hook(save_gradient(captured, name + "_g")))

    model.zero_grad(set_to_none=True)
    logit = model(ecg_tensor, pcg_tensor)
    logit.sum().backward()
    for handle in handles:
        handle.remove()

    cams = {}
    for name in ["ecg", "pcg"]:
        weights = captured[name + "_g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * captured[name + "_a"]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        cam = cam - cam.min()
        if cam.max() > CAM_EPSILON:
            cam = cam / cam.max()
        cams[name] = cam
    return cams


def frequency_ticks(scales, wavelet):
    """Label image rows in Hz. Row 0 is the smallest scale, i.e. the HIGHEST
    frequency - an easy thing to get backwards, so the axis says it outright."""
    positions = np.linspace(0, len(scales) - 1, IMAGE_SIZE)
    row_scales = np.interp(positions, np.arange(len(scales)), scales.astype(float))
    frequencies = pywt.scale2frequency(wavelet, row_scales) * FS
    rows = np.linspace(0, IMAGE_SIZE - 1, N_FREQ_TICKS).astype(int)
    return rows, [str(int(round(frequencies[row]))) for row in rows]


def render(ecg_image, pcg_image, cams, title):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    panels = [
        (0, ecg_image, None, "ECG scalogram", ECG_SCALES, ECG_WAVELET),
        (0, ecg_image, cams["ecg"], "ECG Grad-CAM", ECG_SCALES, ECG_WAVELET),
        (1, pcg_image, None, "PCG scalogram", PCG_SCALES, PCG_WAVELET),
        (1, pcg_image, cams["pcg"], "PCG Grad-CAM", PCG_SCALES, PCG_WAVELET),
    ]
    for index, panel in enumerate(panels):
        row, image, cam, heading, scales, wavelet = panel
        axis = axes[row][index % 2]
        if cam is None:
            axis.imshow(image, cmap="jet", aspect="auto", origin="lower")
        else:
            axis.imshow(image, cmap="gray", aspect="auto", origin="lower")
            axis.imshow(cam, cmap="jet", alpha=OVERLAY_ALPHA, aspect="auto", origin="lower")
        tick_rows, tick_labels = frequency_ticks(scales, wavelet)
        axis.set_yticks(tick_rows)
        axis.set_yticklabels(tick_labels)
        axis.set_ylabel("frequency (Hz)")
        axis.set_xticks(np.linspace(0, IMAGE_SIZE - 1, 4))
        axis.set_xticklabels(["0", "1", "2", "3"])
        axis.set_xlabel("time (s)")
        axis.set_title(heading)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


# ------------------------------------------------------------- analysis


def example_truth(files):
    """Ground-truth label of a bundled example, read from its file name."""
    for path in as_paths(files):
        name = os.path.basename(path)
        if name.startswith("normal_"):
            return "normal"
        if name.startswith("abnormal_"):
            return "abnormal"
    return ""


def threshold_note(decision):
    """State the operating point as it actually is, with its cross-validated rates."""
    threshold = decision["threshold"]
    if "fold_test_sensitivity" not in decision:
        return f"Decision threshold: **{round(threshold, 3)}**.\n\n"
    if threshold < DEFAULT_THRESHOLD:
        relation = "below the default 0.5, set low for screening"
    else:
        relation = "at or above the default 0.5"
    if decision["target_sensitivity"] >= 1.0:
        target_text = "every abnormal validation record"
    else:
        target_text = f"at least {round(100 * decision['target_sensitivity'])}% of abnormal validation records"
    return (
        f"Decision threshold: **{round(threshold, 3)}**, {relation} - the highest threshold still "
        f"catching {target_text}. On this model's held-out test fold it detects "
        f"**{round(100 * decision['fold_test_sensitivity'])}%** of abnormal records and clears "
        f"**{round(100 * decision['fold_test_specificity'])}%** of normal ones; across all five "
        f"cross-validation folds the same procedure averaged "
        f"{round(100 * decision['cv_sensitivity'])}% and {round(100 * decision['cv_specificity'])}%.\n\n"
    )


def analyse(files):
    try:
        ecg, pcg = read_upload(files)
    except (ValueError, OSError) as err:
        return f"### Could not read the input\n\n{err}\n\n{DISCLAIMER}", None

    ecg = fill_non_finite(ecg)
    pcg = fill_non_finite(pcg)
    peak = float(np.max(np.abs(pcg)))
    if peak > 0:
        pcg = pcg / peak

    windows = find_windows(ecg)
    if len(windows) == 0:
        return (
            "### No usable 3-second window\n\nR-peak detection found no full 3 s window "
            "centred on a beat. The recording may be too short (it needs a few seconds "
            f"either side of a beat) or too noisy.\n\n{DISCLAIMER}"
        ), None

    model = STATE["model"]
    decision = STATE["decision"]
    threshold = decision["threshold"]
    images = []
    probabilities = []
    with torch.no_grad():
        for start, stop in windows:
            ecg_image = compute_scalogram(ecg[start:stop], ECG_SCALES, ECG_WAVELET)
            pcg_image = compute_scalogram(pcg[start:stop], PCG_SCALES, PCG_WAVELET)
            ecg_tensor = torch.from_numpy(ecg_image)[None, None]
            pcg_tensor = torch.from_numpy(pcg_image)[None, None]
            probabilities.append(float(torch.sigmoid(model(ecg_tensor, pcg_tensor))[0]))
            images.append((ecg_image, pcg_image))

    record_probability = float(np.mean(probabilities))
    if record_probability >= threshold:
        label = "abnormal"
        shown = int(np.argmax(probabilities))
    else:
        label = "normal"
        shown = int(np.argmin(probabilities))

    ecg_image, pcg_image = images[shown]
    cams = grad_cam_pair(
        model, torch.from_numpy(ecg_image)[None, None], torch.from_numpy(pcg_image)[None, None]
    )
    start_s = round(windows[shown][0] / FS, 1)
    title = f"window {shown + 1} of {len(windows)} (starts at {start_s} s), p(abnormal) = {round(probabilities[shown], 3)}"
    figure = render(ecg_image, pcg_image, cams, title)

    per_window = ", ".join([str(round(p, 3)) for p in probabilities])

    # The examples were chosen by record ID before anyone looked at the model's
    # output, so some are misclassified. Say so, rather than let a wrong answer
    # on a bundled example pass as a right one.
    truth_note = ""
    truth = example_truth(files)
    if truth != "":
        if truth == label:
            truth_note = f"Ground truth for this example record: **{truth}** - the prediction is correct.\n\n"
        else:
            truth_note = (
                f"Ground truth for this example record: **{truth}** - **the prediction is wrong.** "
                "This model's most common error is calling normal records abnormal, and a "
                "screening threshold makes that error more common on purpose.\n\n"
            )

    summary = (
        f"## Prediction: **{label}**\n\n"
        f"Record-level P(abnormal) = **{round(record_probability, 3)}** - the mean over "
        f"{len(windows)} beat-centred 3 s windows.\n\n"
        + threshold_note(decision)
        + f"Per-window probabilities: {per_window}\n\n"
        + truth_note
        + "The heatmap shows the window that most supports the prediction. Grad-CAM here "
        "comes from a 7 x 7 feature map, so it is coarse by construction.\n\n"
        + DISCLAIMER
    )
    return summary, figure


def example_inputs():
    if not os.path.isdir(EXAMPLE_DIR):
        return []
    names = sorted([name for name in os.listdir(EXAMPLE_DIR) if name.endswith(".csv")])
    return [[[os.path.join(EXAMPLE_DIR, name)]] for name in names]


def build_interface():
    with gr.Blocks(title="ECG-PCG Cardiac Abnormality Detection") as demo:
        gr.Markdown(
            "# ECG-PCG Cardiac Abnormality Detection\n\n"
            "Upload a **synchronous ECG + PCG recording at 2000 Hz** - either a WFDB "
            "record (select the `.hea` and `.dat` together) or a CSV with a header row "
            "`ecg,pcg`. The model is a dual-branch pretrained ResNet-18 with bidirectional "
            "cross-modal attention over CWT scalograms, trained on PhysioNet/CinC 2016 "
            "Training-A with **record-level (patient-level) splits**.\n\n" + DISCLAIMER
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="ECG + PCG recording", file_count="multiple",
                    file_types=[".csv", ".hea", ".dat"], type="filepath",
                )
                button = gr.Button("Analyse", variant="primary")
                examples = example_inputs()
                if len(examples) > 0:
                    gr.Examples(
                        examples=examples, inputs=[upload],
                        label="Example records - from the held-out test fold of the deployed model",
                    )
            with gr.Column(scale=2):
                summary = gr.Markdown()
                plot = gr.Plot(label="Scalograms and Grad-CAM")
        button.click(analyse, inputs=[upload], outputs=[summary, plot])
    return demo


def main():
    STATE["model"] = load_model()
    STATE["decision"] = load_decision()
    build_interface().launch()


if __name__ == "__main__":
    main()
