"""Gradio demo: upload ECG + PCG -> prediction, confidence and Grad-CAM.

Accepts either a WFDB record (.hea plus .dat) or a two-column CSV of raw
samples. Takes the first 3 seconds at 2000 Hz, which is what the model was
trained on.

Self-contained for the same reason as the training scripts and the FastAPI app:
a Spaces deployment ships this file plus a checkpoint.

Deploying to HuggingFace Spaces is documented in docs/logs/tasks/5-mlops.md but
not attempted here - no token was provided.
"""

import os

import gradio as gr
import matplotlib
import numpy as np
import pywt
import torch
import torch.nn as nn
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SIZE = 224
UINT8_MAX = 255.0
EXPECTED_FS = 2000
WINDOW_SECONDS = 3.0
EXPECTED_SAMPLES = int(EXPECTED_FS * WINDOW_SECONDS)

ECG_WAVELET = "cmor1.5-1.0"
ECG_SCALES = np.arange(20, 501)
PCG_WAVELET = "morl"
PCG_SCALES = np.arange(7, 131)

DROPOUT = 0.5
N_HEADS = 4
CAM_EPSILON = 1e-8
# Must equal PAD_SCALE_FACTOR in scripts/features/01_scalogram.py.
PAD_SCALE_FACTOR = 4

CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT", "models/cross_attn_fusion/default/default_cv_fold0_best.pth"
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def feature_maps(self, x):
        return self.features(x)

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class BidirectionalCrossAttention(nn.Module):
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


class CrossAttnFusion(nn.Module):
    def __init__(self, dropout, n_heads):
        super().__init__()
        self.ecg_branch = CNNBranch()
        self.pcg_branch = CNNBranch()
        self.cross_attention = BidirectionalCrossAttention(256, n_heads)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
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
    model = CrossAttnFusion(DROPOUT, N_HEADS)
    loaded = False
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        loaded = True
    model.eval()
    return model, loaded


MODEL, MODEL_LOADED = load_model()


def resize_to_square(field, size):
    return np.asarray(Image.fromarray(field, mode="F").resize((size, size), Image.BILINEAR))


def compute_scalogram(signal, scales, wavelet, fs):
    # Reflect-pad before transforming, then crop back - IDENTICAL to
    # scripts/features/01_scalogram.py. Without this the ECG scalogram is
    # dominated by the CWT cone-of-influence artifact that training removed, and
    # the model is fed a distribution it never saw (measured: mean |diff| ~59/255
    # per ECG pixel against the training memmap).
    pad = int(min(len(signal), PAD_SCALE_FACTOR * int(np.max(scales))))
    padded = np.pad(signal, pad, mode="reflect") if pad > 0 else signal
    coeffs = pywt.cwt(padded, scales, wavelet, sampling_period=1.0 / fs, method="fft")[0]
    magnitude = np.abs(coeffs).astype(np.float32)
    if pad > 0:
        magnitude = magnitude[:, pad : pad + len(signal)]
    resized = resize_to_square(magnitude, IMAGE_SIZE)
    low = float(np.min(resized))
    high = float(np.max(resized))
    if high - low <= 0:
        return np.zeros(resized.shape, dtype=np.float32)
    quantised = np.clip(((resized - low) / (high - low)) * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)
    return quantised.astype(np.float32) / UINT8_MAX


def read_signals(record_file, csv_file):
    """Return (ecg, pcg) or raise ValueError with something a user can act on."""
    if record_file is not None:
        import wfdb

        path = record_file.name if hasattr(record_file, "name") else record_file
        base = os.path.splitext(path)[0]
        record = wfdb.rdrecord(base)
        names = [name.strip().upper() for name in record.sig_name]
        if "ECG" not in names or "PCG" not in names:
            raise ValueError(f"record needs both ECG and PCG channels, found {record.sig_name}")
        ecg = record.p_signal[:, names.index("ECG")].astype(np.float32)
        pcg = record.p_signal[:, names.index("PCG")].astype(np.float32)
        return ecg, pcg, int(record.fs)

    if csv_file is not None:
        path = csv_file.name if hasattr(csv_file, "name") else csv_file
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("CSV must have two columns: ecg,pcg")
        return data[:, 0], data[:, 1], EXPECTED_FS

    raise ValueError("upload a WFDB record or a two-column CSV")


def grad_cam(model, branch, ecg_tensor, pcg_tensor):
    convs = [layer for layer in branch.features.modules() if isinstance(layer, nn.Conv2d)]
    captured = {}

    forward_handle = convs[-1].register_forward_hook(
        lambda module, layer_input, layer_output: captured.__setitem__("a", layer_output.detach())
    )
    backward_handle = convs[-1].register_full_backward_hook(
        lambda module, grad_input, grad_output: captured.__setitem__("g", grad_output[0].detach())
    )

    model.zero_grad(set_to_none=True)
    logit = model(ecg_tensor, pcg_tensor)
    logit.backward()
    forward_handle.remove()
    backward_handle.remove()

    weights = captured["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * captured["a"]).sum(dim=1))
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > CAM_EPSILON:
        cam = cam / cam.max()
    return cam, float(torch.sigmoid(logit.detach())[0])


def analyse(record_file, csv_file):
    try:
        ecg, pcg, fs = read_signals(record_file, csv_file)
    except (ValueError, OSError) as err:
        return f"Could not read the input: {err}", None

    if fs != EXPECTED_FS:
        return f"This model expects {EXPECTED_FS} Hz, the file is {fs} Hz.", None
    if len(ecg) < EXPECTED_SAMPLES:
        return f"Need at least 3 s ({EXPECTED_SAMPLES} samples), got {len(ecg)}.", None

    ecg = ecg[:EXPECTED_SAMPLES]
    pcg = pcg[:EXPECTED_SAMPLES]
    peak = float(np.max(np.abs(pcg)))
    if peak > 0:
        pcg = pcg / peak

    ecg_image = compute_scalogram(ecg, ECG_SCALES, ECG_WAVELET, fs)
    pcg_image = compute_scalogram(pcg, PCG_SCALES, PCG_WAVELET, fs)
    ecg_tensor = torch.from_numpy(ecg_image).unsqueeze(0).unsqueeze(0)
    pcg_tensor = torch.from_numpy(pcg_image).unsqueeze(0).unsqueeze(0)

    ecg_cam, probability = grad_cam(MODEL, MODEL.ecg_branch, ecg_tensor, pcg_tensor)
    pcg_cam, _ = grad_cam(MODEL, MODEL.pcg_branch, ecg_tensor, pcg_tensor)

    label = "abnormal" if probability >= 0.5 else "normal"
    confidence = probability if probability >= 0.5 else 1.0 - probability

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0][0].imshow(ecg_image, cmap="jet", aspect="auto", origin="lower")
    axes[0][0].set_title("ECG scalogram")
    axes[0][1].imshow(ecg_image, cmap="gray", aspect="auto", origin="lower")
    axes[0][1].imshow(ecg_cam, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    axes[0][1].set_title("ECG Grad-CAM")
    axes[1][0].imshow(pcg_image, cmap="jet", aspect="auto", origin="lower")
    axes[1][0].set_title("PCG scalogram")
    axes[1][1].imshow(pcg_image, cmap="gray", aspect="auto", origin="lower")
    axes[1][1].imshow(pcg_cam, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    axes[1][1].set_title("PCG Grad-CAM")
    figure.tight_layout()

    warning = "" if MODEL_LOADED else "\n\nWARNING: no checkpoint found, this is an UNTRAINED model."
    summary = (
        f"Prediction: **{label}**\n\n"
        f"P(abnormal) = {round(probability, 4)}\n\n"
        f"Confidence = {round(confidence, 4)}" + warning
    )
    return summary, figure


def build_interface():
    return gr.Interface(
        fn=analyse,
        inputs=[
            gr.File(label="WFDB record (.hea, upload .dat alongside)"),
            gr.File(label="or a two-column CSV: ecg,pcg at 2000 Hz"),
        ],
        outputs=[gr.Markdown(label="Prediction"), gr.Plot(label="Scalograms and Grad-CAM")],
        title="ECG-PCG Cardiac Abnormality Detection",
        description=(
            "Dual-branch CWT scalogram fusion with cross-modal attention, trained on "
            "PhysioNet/CinC 2016 Training-A with record-level (patient-level) splits. "
            "Takes the first 3 s at 2000 Hz. Research demo - not a medical device."
        ),
    )


if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
