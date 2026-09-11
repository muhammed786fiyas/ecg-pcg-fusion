"""FastAPI service for the ECG-PCG fusion model.

Serves cross_attn_resnet18, the study's best family (patient AUC 0.938 +/- 0.039
over record-level 5-fold CV): a pretrained ResNet-18 per modality with
bidirectional cross-modal attention. It scores ONE 3 s window per request; the
local Gradio demo (hf_space_app.py) scores whole recordings. Until 2026-09-11
this service carried the custom-CNN cross_attn_fusion model instead.

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
import torchvision
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

MODEL_VERSION = "cross_attn_resnet18/1"
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
PROJ_DIM = 256
CAM_EPSILON = 1e-8
# Must equal PAD_SCALE_FACTOR in scripts/features/01_scalogram.py.
PAD_SCALE_FACTOR = 4

# Fold 0: the CV model with the highest inner-VALIDATION AUC (0.9745), the same
# weights the Gradio demo serves. Chosen without looking at test performance.
CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT", "models/cross_attn_resnet18/default/default_cv_fold0_best.pth"
)
PARAMS_PATH = os.environ.get("PARAMS_PATH", "params.yaml")


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
      holding a state dict. It is loaded into an eager CrossAttnResNet18, which
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
        model = CrossAttnResNet18(DROPOUT, N_HEADS, PROJ_DIM)
        model.eval()
        STATE["model"] = model
        STATE["supports_explain"] = True
        print(f"WARNING: no model at {CHECKPOINT_PATH}, serving an untrained model")
        return model

    loaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "model" in loaded:
        model = CrossAttnResNet18(DROPOUT, N_HEADS, PROJ_DIM)
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
    description="Dual-branch pretrained ResNet-18 over CWT scalograms, with cross-modal attention.",
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
    # The last Conv2d of the branch - for ResNetBranch, the 1x1 projection on
    # layer4's 7x7 map, the same layer the Gradio demo and 02_gradcam.py hook.
    convs = [layer for layer in branch.modules() if isinstance(layer, nn.Conv2d)]
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
