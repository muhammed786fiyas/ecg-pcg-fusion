# Docker

Two images, built from the repo root (not from this folder).

## Training image — CPU only, deliberately

```
docker build -f docker/Dockerfile.train -t ecg-pcg-train .
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/models:/workspace/models" \
  ecg-pcg-train python scripts/modeling/dual_cnn/01_train.py --protocol dev
```

It uses a **CPU-only PyTorch base**. The development machine has no CUDA GPU, and
an image that assumes one would fail confusingly. Real training runs on Kaggle's
free GPU through `scripts/remote/`, using the same training scripts unchanged.

## Inference image

Export the TorchScript model first — the image carries only that, not a
checkpoint plus the training code:

```
python docker/export_torchscript.py \
  --checkpoint models/cross_attn_fusion/default/default_cv_fold0_best.pth
docker build -f docker/Dockerfile.inference -t ecg-pcg-serve .
docker run --rm -p 8000:8000 ecg-pcg-serve
```

Then `GET /health`, `POST /predict`, `POST /explain`.

`/health` reports `degraded` rather than crash-looping when no checkpoint is
present, so the container can tell you what is actually wrong.
