# ECG–PCG Cardiac Abnormality Detection

[![CI](https://github.com/muhammed786fiyas/ecg-pcg-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammed786fiyas/ecg-pcg-fusion/actions/workflows/ci.yml)

Dual-branch CWT scalogram fusion with cross-modal attention, on PhysioNet/CinC
2016 Training-A (405 records with synchronous ECG + PCG).

Target venue: *Biomedical Signal Processing and Control* (Elsevier).

---

## The one thing to know

**The patient is the unit of splitting.** Every split — train, val, test, every
CV fold — is decided over *record IDs*, never over segments or augmented rows.
`scripts/data/05_assign_folds.py` receives only `(record_id, label)` pairs and a
seed, so a segment-derived quantity cannot structurally reach it, and
`tests/test_leakage.py` + `tests/test_split_inputs.py` assert that rather than
trusting it.

This matters because published PhysioNet 2016 accuracies of 93–99% largely
predate leakage-controlled protocols. Contribution 4 below measures how much of
that gap is manufactured by split choice.

## Contributions

1. **CWT mother-wavelet sensitivity** — each modality's wavelet varied over
   `{cmor1.5-1.0, morl, mexh, gaus4}` with the other held at default, scales
   fixed, `dual_cnn` throughout so only the input representation varies.
2. **Grad-CAM physiological validation** — does ECG attention land on the QRS
   time–frequency region and PCG attention on the S1/S2 bursts?
3. **Segment-to-patient aggregation** — majority vote vs mean probability vs max
   probability, at patient level.
4. **Split-protocol sensitivity, a deliberate negative control** — the same model
   trained under a correct record-level split, a leaky segment-level validation
   split, and a fully leaky split. Labelled a negative control everywhere it
   appears.

## Layout

```
params.yaml            tunable knobs only, nested <module>.<substage>
dvc.yaml               every file path lives here, not in params.yaml
CLAUDE.md              coding conventions + the split rules
PROJECT_CONTEXT.md     protocol, decisions and their history, citation map
scripts/
  data/                01_convert .. 06_augment
  features/            01_scalogram  (CWT -> uint8 memmap)
  modeling/            one self-contained training script per family
  evaluation/          patient aggregation, Grad-CAM, tables, figures
  remote/              Kaggle GPU offload: sync, template, run, merge
  serving/             FastAPI app + Gradio demo
tests/                 leakage guards first, then shapes and contracts
docs/logs/             daily/ and per-module tasks/
```

## Running it

The project uses a dedicated conda env named `ecg_pcg` (Python 3.11). Each shell
is fresh, so `conda activate` does not persist — prefix every command:

```
conda run -n ecg_pcg python -m pip install -r requirements.txt
conda run -n ecg_pcg dvc repro
conda run -n ecg_pcg python -m pytest tests/ -q
```

Use `python -m pip`, never bare `pip` — the `pip.exe` launcher fails with
"Access is denied" on the development machine.

### Compute split

The data pipeline runs locally on CPU. **Training runs on Kaggle's free GPU**,
because it measured at ~500 s/epoch on this CPU — roughly 120 h for 5-fold CV
across all seven model families, which does not finish. Training code is
device-agnostic (AMP only under `torch.cuda.is_available()`), so the identical
script smoke-tests locally and runs on a Kaggle T4:

```
conda run -n ecg_pcg python scripts/remote/01_sync_dataset.py
conda run -n ecg_pcg python scripts/remote/02_make_kernel.py --family dual_cnn --fold 0
conda run -n ecg_pcg python scripts/remote/03_run_kernel.py --job-dir .kaggle_kernels/<slug>
conda run -n ecg_pcg python scripts/remote/04_merge_results.py --kernel-output <slug>/output
```

## Serving

**Browser demo (local)** — the strongest model, `cross_attn_resnet18`, scoring
a whole recording over beat-centred 3 s windows, with Grad-CAM in Hz:

```
conda run -n ecg_pcg python scripts/serving/build_hf_space.py   # assemble .hf_space_staging/
cd .hf_space_staging
conda run -n ecg_pcg python app.py                             # http://localhost:7860
```

The decision threshold is 0.5. A lower, validation-fitted screening threshold
was evaluated and not adopted - see
`reports/screening_threshold/cross_attn_resnet18/summary.md` after running
`scripts/evaluation/06_screening_threshold.py`. The demo is local only; there is
no hosted public version.

**REST API**:

```
conda run -n ecg_pcg uvicorn app:app --app-dir scripts/serving   # /health /predict /explain
```

See [docker/README.md](docker/README.md) for the CPU-only training image and the
TorchScript inference image.

## Reproducibility

A single `global_seed` in `params.yaml`. Augmentation is seeded **per (segment,
variant)** from a stable hash, never from a global RNG stream — a global stream
would make output depend on filesystem iteration order, which is a large part of
why the previous pipeline was rebuilt from scratch.
`tests/test_reproducibility.py` asserts that re-deriving any variant from its
seed reproduces the stored file byte-for-byte.

---

Research code. **Not a medical device.**
