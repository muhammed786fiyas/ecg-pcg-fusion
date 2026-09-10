"""FastAPI service for the cross-attention ECG-PCG fusion model.

Endpoints:
  GET  /health   model loaded, version, device
  POST /predict  raw 3 s ECG + PCG at 2000 Hz -> class + confidence
  POST /explain  same input -> Grad-CAM PNG over both scalograms

Self-contained, like the training scripts: the model definitions and the
preprocessing live here rather than being imported from a sibling, so the
inference image can ship this one file plus a checkpoint.

The preprocessing must match training exactly - same wavelets, same scales, same
per-image uint8 normalization - or the model sees a distribution it never saw in
training. The constants below are duplicated from params.yaml for that reason and
are checked against it at startup when params.yaml is present.
"""

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import pywt
import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

MODEL_VERSION = "cross_attn_fusion/1"
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

CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT", "models/cross_attn_fusion/default/default_cv_fold0_best.pth"
)
PARAMS_PATH = os.environ.get("PARAMS_PATH", "params.yaml")


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


def resize_to_square(field, size):
    return np.asarray(Image.fromarray(field, mode="F").resize((size, size), Image.BILINEAR))


def scalogram_to_unit(field):
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low <= 0:
        return np.zeros(field.shape, dtype=np.float32)
    quantised = np.clip(((field - low) / (high - low)) * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)
    return quantised.astype(np.float32) / UINT8_MAX


def compute_scalogram(signal, scales, wavelet, fs):
    """Identical to the training-time transform, including the uint8 round trip.

    The quantisation is not cosmetic: training saw uint8 scalograms, so inference
    must quantise too or it feeds the model a slightly different distribution.
    """
    coeffs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs, method="fft")[0]
    magnitude = np.abs(coeffs).astype(np.float32)
    return scalogram_to_unit(resize_to_square(magnitude, IMAGE_SIZE))


def normalize_pcg(pcg):
    peak = float(np.max(np.abs(pcg)))
    if peak <= 0 or not np.isfinite(peak):
        return pcg
    return (pcg / peak).astype(np.float32)


def check_preprocessing_matches_params():
    """Guard against the service and the pipeline drifting apart."""
    if not os.path.exists(PARAMS_PATH):
        return "params.yaml not present, preprocessing constants unverified"
    with open(PARAMS_PATH) as handle:
        params = yaml.safe_load(handle)
    feature_params = params["features"]["scalogram"]
    mismatches = []
    if feature_params["ecg_wavelet"] != ECG_WAVELET:
        mismatches.append("ecg_wavelet")
    if feature_params["pcg_wavelet"] != PCG_WAVELET:
        mismatches.append("pcg_wavelet")
    if feature_params["image_size"] != IMAGE_SIZE:
        mismatches.append("image_size")
    if int(feature_params["ecg_scale_start"]) != int(ECG_SCALES[0]):
        mismatches.append("ecg_scale_start")
    if int(feature_params["pcg_scale_start"]) != int(PCG_SCALES[0]):
        mismatches.append("pcg_scale_start")
    if len(mismatches) > 0:
        return "MISMATCH with params.yaml: " + ", ".join(mismatches)
    return "matches params.yaml"


class SignalRequest(BaseModel):
    ecg: list = Field(..., description=f"{EXPECTED_SAMPLES} float samples, 3 s at {EXPECTED_FS} Hz")
    pcg: list = Field(..., description=f"{EXPECTED_SAMPLES} float samples, 3 s at {EXPECTED_FS} Hz")
    fs: int = Field(EXPECTED_FS, description="sampling rate in Hz")


class PredictResponse(BaseModel):
    label: str
    label_index: int
    probability_abnormal: float
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supports_explain: bool
    model_version: str
    checkpoint: str
    device: str
    preprocessing: str


STATE = {"model": None, "device": torch.device("cpu"), "loaded": False, "supports_explain": False}


def load_model():
    """Load either a training checkpoint or a TorchScript export.

    Both are legitimate inputs and they need different handling:

    - a **training checkpoint** (`*_best.pth`) is a dict with a "model" key
      holding a state dict. It is loaded into an eager CrossAttnFusion, which
      supports Grad-CAM because hooks can be registered on its submodules.
    - a **TorchScript export** (`*.pt`) deserialises to a RecursiveScriptModule.
      It runs /predict fine but cannot serve /explain: registering backward
      hooks on a scripted module's internals is not supported.

    The first version assumed the checkpoint form and did `checkpoint["model"]`
    unconditionally. Against the TorchScript artifact the Docker image actually
    ships, that raises NotImplementedError and the container dies at startup.
    The local tests never caught it because they point at the .pth.
    """
    STATE["supports_explain"] = False
    if not os.path.exists(CHECKPOINT_PATH):
        # Serve with random weights rather than refusing to start, so /health can
        # report the real problem instead of the container crash-looping.
        STATE["loaded"] = False
        model = CrossAttnFusion(DROPOUT, N_HEADS)
        model.eval()
        STATE["model"] = model
        STATE["supports_explain"] = True
        print(f"WARNING: no model at {CHECKPOINT_PATH}, serving an untrained model")
        return model

    loaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "model" in loaded:
        model = CrossAttnFusion(DROPOUT, N_HEADS)
        model.load_state_dict(loaded["model"])
        STATE["supports_explain"] = True
        print(f"loaded training checkpoint {CHECKPOINT_PATH} (Grad-CAM available)")
    else:
        model = loaded
        print(f"loaded TorchScript module {CHECKPOINT_PATH} (Grad-CAM unavailable)")

    model.eval()
    STATE["loaded"] = True
    STATE["model"] = model
    return model


@asynccontextmanager
async def lifespan(application):
    load_model()
    yield


app = FastAPI(
    title="ECG-PCG Cardiac Abnormality Detection",
    description="Dual-branch CWT scalogram fusion with cross-modal attention.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def validate(request):
    if len(request.ecg) != EXPECTED_SAMPLES or len(request.pcg) != EXPECTED_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"expected {EXPECTED_SAMPLES} samples per channel "
                f"(3 s at {EXPECTED_FS} Hz), got ecg={len(request.ecg)} pcg={len(request.pcg)}"
            ),
        )
    if request.fs != EXPECTED_FS:
        raise HTTPException(status_code=422, detail=f"expected fs={EXPECTED_FS}, got {request.fs}")

    ecg = np.asarray(request.ecg, dtype=np.float32)
    pcg = np.asarray(request.pcg, dtype=np.float32)
    if not np.isfinite(ecg).all() or not np.isfinite(pcg).all():
        raise HTTPException(status_code=422, detail="input contains NaN or Inf")
    return ecg, normalize_pcg(pcg)


def to_tensors(ecg, pcg, fs):
    ecg_image = compute_scalogram(ecg, ECG_SCALES, ECG_WAVELET, fs)
    pcg_image = compute_scalogram(pcg, PCG_SCALES, PCG_WAVELET, fs)
    ecg_tensor = torch.from_numpy(ecg_image).unsqueeze(0).unsqueeze(0)
    pcg_tensor = torch.from_numpy(pcg_image).unsqueeze(0).unsqueeze(0)
    return ecg_image, pcg_image, ecg_tensor, pcg_tensor


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if STATE["loaded"] else "degraded",
        model_loaded=STATE["loaded"],
        supports_explain=bool(STATE["supports_explain"]),
        model_version=MODEL_VERSION,
        checkpoint=CHECKPOINT_PATH,
        device=str(STATE["device"]),
        preprocessing=check_preprocessing_matches_params(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: SignalRequest):
    ecg, pcg = validate(request)
    _, _, ecg_tensor, pcg_tensor = to_tensors(ecg, pcg, request.fs)

    with torch.no_grad():
        logit = STATE["model"](ecg_tensor, pcg_tensor)
    probability = float(torch.sigmoid(logit)[0])
    label_index = 1 if probability >= 0.5 else 0

    return PredictResponse(
        label="abnormal" if label_index == 1 else "normal",
        label_index=label_index,
        probability_abnormal=probability,
        confidence=probability if label_index == 1 else 1.0 - probability,
        model_version=MODEL_VERSION,
    )


def grad_cam_for_branch(model, branch, ecg_tensor, pcg_tensor):
    convs = [layer for layer in branch.features.modules() if isinstance(layer, nn.Conv2d)]
    target = convs[-1]

    captured = {}

    def forward_hook(module, layer_input, layer_output):
        captured["activations"] = layer_output.detach()

    def backward_hook(module, grad_input, grad_output):
        captured["gradients"] = grad_output[0].detach()

    forward_handle = target.register_forward_hook(forward_hook)
    backward_handle = target.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logit = model(ecg_tensor, pcg_tensor)
    logit.backward()

    forward_handle.remove()
    backward_handle.remove()

    weights = captured["gradients"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * captured["activations"]).sum(dim=1))
    resized = torch.nn.functional.interpolate(
        cam.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    cam = resized[0, 0].cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > CAM_EPSILON:
        cam = cam / cam.max()
    return cam, float(torch.sigmoid(logit.detach())[0])


@app.post("/explain")
def explain(request: SignalRequest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not STATE["supports_explain"]:
        raise HTTPException(
            status_code=501,
            detail=(
                "Grad-CAM needs an eager model, but this service loaded a TorchScript "
                "export, whose internals cannot take backward hooks. Point "
                "MODEL_CHECKPOINT at a training checkpoint (*_best.pth) to enable /explain."
            ),
        )

    ecg, pcg = validate(request)
    ecg_image, pcg_image, ecg_tensor, pcg_tensor = to_tensors(ecg, pcg, request.fs)

    model = STATE["model"]
    ecg_cam, probability = grad_cam_for_branch(model, model.ecg_branch, ecg_tensor, pcg_tensor)
    pcg_cam, _ = grad_cam_for_branch(model, model.pcg_branch, ecg_tensor, pcg_tensor)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(ecg_image, cmap="gray", aspect="auto", origin="lower")
    axes[0].imshow(ecg_cam, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    axes[0].set_title("ECG Grad-CAM")
    axes[1].imshow(pcg_image, cmap="gray", aspect="auto", origin="lower")
    axes[1].imshow(pcg_cam, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    axes[1].set_title("PCG Grad-CAM")
    for axis in axes:
        axis.set_xlabel("time (3 s window)")
        axis.set_ylabel("scale")
    label_name = "abnormal" if probability >= 0.5 else "normal"
    figure.suptitle(f"prediction: {label_name} (p={round(probability, 3)}) - {MODEL_VERSION}")
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130)
    plt.close(figure)
    return Response(content=buffer.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
