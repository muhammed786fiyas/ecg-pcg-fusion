# Task log 5 — MLOps

## Scope
DVC pipeline, MLflow tracking, Kaggle GPU offload, Docker, FastAPI, GitHub
Actions CI, Gradio demo, and the Kaggle GPU-quota ledger.

## Completed
- **Environment recorded.** `conda run -n ecg_pcg python --version` reports
  **Python 3.11.16**. Per §0.5 of the brief this is the 3.10+ branch, so current
  versions of `mlflow`, `gradio` and `numpy` were taken rather than the 3.9
  fallback pins.
  - One pin was still required, for a different reason: **`numpy<2`**, because
    `neurokit2==0.2.7` (pinned by the brief) predates the NumPy 2 ABI break.
    This is a consequence of the neurokit2 pin, not of the Python version.
- `.env` written with `MLFLOW_TRACKING_URI=file:./models/mlflow_tracking`,
  `MLFLOW_ALLOW_FILE_STORE=true`, and the Kaggle username/dataset slug.
  `.env.example` committed as the template; `.env` itself is gitignored.
- Kaggle CLI 2.2.4 authenticated via `~/.kaggle/credentials.json` (OAuth, from
  `kaggle auth login`). Verified with `kaggle datasets list`.
  Kaggle username: `muhammed786fiyas`.

## Key decisions
- **MLflow file store, not a server.** No tracking server to keep alive, and the
  run directories are plain files — which is what makes §9's
  `04_merge_results.py` able to merge Kaggle-returned runs into the local store
  by copying directories.
- **`MLFLOW_ALLOW_FILE_STORE` is set in-script** via `os.environ.setdefault(...)`,
  not only in `.env`. MLflow 3.x refuses the plain filesystem backend otherwise,
  and a Kaggle kernel has no `.env` to read.
- **One kernel per (model config, fold).** Keeps every job far from Kaggle's
  9-hour session ceiling and makes a failure cost one fold rather than a sweep.

## Measured CPU throughput — why training goes to Kaggle

Timed on the real data, `dual_cnn`, CV fold 0 (10 104 train rows / 470 val /
756 test), 16 CPU threads:

> **1130 s wall clock for 2 epochs plus a test pass — roughly 500 s per epoch.**

Extrapolating at ~25 epochs to early stopping: ~3.5 h per fold, ~17 h per model
family, and **~120 h for 5-fold CV across all seven families** — before the
wavelet ablation. Local CPU training is not a fallback that costs a bit more
time; it does not finish. The GPU offload is load-bearing.

## Kaggle gotchas — all found the hard way in the dry run

1. **The dataset is mounted at `/kaggle/input/datasets/<owner>/<slug>`**, not
   `/kaggle/input/<slug>`. The entry script now *discovers* the dataset root by
   searching for the payload marker rather than hardcoding either layout.
2. **A kernel pushed before the dataset version finishes processing sees no
   dataset at all.** Wait until `kaggle datasets files <slug>` lists the
   scalograms before pushing.
3. **`mlflow` is not in the Kaggle image.** The entry script pip-installs
   `mlflow` and `python-dotenv`, which is why `enable_internet: "true"` is
   mandatory in `kernel-metadata.json`.
4. **Kaggle auto-extracts an uploaded `.zip` into the dataset**, so the payload
   arrives as a directory tree. The entry script still handles the still-zipped
   case, since that behaviour is not contractual.
5. **`kaggle datasets create` defaults to `--dir-mode skip`, which silently skips
   subdirectories.** Everything therefore goes up as one `payload.zip`.
6. **Resume-safety does not survive across kernel invocations.**
   `/kaggle/working` is fresh on every run, so checkpoints do not persist between
   them. This is the real argument for the brief's one-kernel-per-(config, fold)
   sizing: a failure costs one fold, and there is no cross-run resume to lean on.

## Data notes & gotchas
- `pip.exe` fails on this machine with "Access is denied" — always use
  `python -m pip`.
- `conda activate` does not persist between tool calls; every Python command is
  prefixed `conda run -n ecg_pcg`.
- **`conda run` crashes** (an unhandled conda plugin error) when its stdout is
  piped to another command and the output is large. It killed a dataset upload
  midway, leaving an empty dataset registered on Kaggle. For long or noisy
  commands, call the env's interpreter directly:
  `C:/Users/muham/.conda/envs/ecg_pcg/python.exe` and
  `.../envs/ecg_pcg/Scripts/kaggle.exe`.
- A JSON body cannot carry a literal `NaN`, so the API's non-finite guard is
  exercised with a float64 (`1e39`) that overflows to `inf` on the cast to
  float32 — the reachable path, rather than an unreachable one.

## Kaggle GPU quota ledger

Weekly budget 30 h; stop launching new jobs at 25 h
(`compute.kaggle_stop_at_hours`). The live ledger is
`docs/logs/kaggle_quota_ledger.csv`, written by `05_run_queue.py`.

### Charge GPU time, not wall clock — a correction

The first version of the ledger recorded wall clock between pushing a kernel and
seeing a terminal status. **That is the wrong quantity.** It includes however
long the job sat in Kaggle's scheduling queue, and Kaggle's 30 h/week budget is
GPU *session* time.

Measured on `ecgpcg-ecg-only-default-cv-fold0`:

| | |
|---|---|
| wall clock, push to terminal | **35.6 min** |
| true in-kernel runtime | **7.9 min** |
| over-counting factor | **4.5×** |

Charging queue wait against the quota would have halted the run at roughly a
fifth of the real allowance, and would have put a badly wrong compute figure in
the paper. `kernel_gpu_seconds()` now reads the true runtime from the final
timestamp of the returned Kaggle log, falling back to wall clock when the log is
unavailable — conservative, which is the right direction for a budget. The
ledger records both columns.

**Note:** the queue running at the time of this fix still used the old
accounting, so its rows over-report. Reconcile them from the fetched logs in
`.kaggle_kernels/*/output/*.log` before quoting a total.

### A fetched fold that never merged — a silent hole in the results

`kaggle kernels output` has been observed **returning a non-zero exit code while
delivering every file**. The queue treated that as a failed fetch and skipped
the merge, so a completed fold's checkpoints and MLflow run sat on disk, absent
from the local store, with nothing failing loudly. A results table built at that
moment would have shown a "5-fold mean" over 4 folds.

It correlates with download size: it has only happened on
`cross_attn_resnet18`, whose 23 M-parameter checkpoints are ~90 MB (best) and
~280 MB (last), an order of magnitude above the custom-CNN families.

Two fixes:
- `collect()` in `05_run_queue.py` now merges **regardless** of the fetch exit
  code. The merge is the real test of whether the fetch worked, since it
  hard-fails when there are no runs.
- `scripts/remote/06_reconcile.py` sweeps every kernel output directory for runs
  missing from the local store and merges them, then **prints which
  (family, fold) pairs are still absent**. Merging is idempotent, so it is
  always safe to re-run.

**Run `06_reconcile.py` before building any results table.** An incomplete family
averaged as if complete is exactly the kind of error a table cannot show you.

### Day 2: the served model was fed the wrong scalograms

The CWT reflect-padding fix went into the training pipeline on day 1, but the
serving code keeps its own copy of the transform (self-contained by convention)
and was never updated. Compared with the training memmap on the same segments,
the served ECG scalograms differed by ~56-62/255 per pixel on average (max ~240).
Day 1's container smoke test still passed, because all six segments it used were
strongly abnormal - so "validated 6/6" was an overstatement.

- Fixed in `scripts/serving/app.py` and `scripts/serving/gradio_demo.py`.
- `tests/test_serving_parity.py` compares the serving transform against the
  training memmap for real segments, within one uint8 level. It needs the
  pipeline outputs, so it skips in CI - **run it locally before any deploy**.
- Container smoke tests must sample **both classes**.
- The Docker rebuild with the fix failed because Docker Desktop was not running;
  `ecg-pcg-serve:latest` stays stale until it is rebuilt.

### Day 2: public HuggingFace Space for cross_attn_resnet18

- `scripts/serving/hf_space_app.py` - the Space's `app.py`.
- `scripts/serving/build_hf_space.py` - stages `.hf_space_staging/` (gitignored:
  it carries an 89 MB checkpoint) and uploads it with `--upload`, reading the
  token from `HF_TOKEN` or a prior `hf auth login`, never printing it.
- Weights selected by inner-validation AUC (fold 0, 0.9745); examples from fold
  0's held-out test partition, chosen by record ID before looking at predictions.
  The normal example is misclassified and the demo says so.
- Requirements pull **CPU** torch wheels via `--extra-index-url`, so the Space
  does not install ~2 GB of CUDA libraries it cannot use.

### Day 2: HuggingFace now charges for Gradio Spaces

`create_repo` returned **402 Payment Required**: "Static Spaces are free for
everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription." Nothing was created. `build_hf_space.py` now turns this into a
clear `QC FAIL`. A static Space cannot run PyTorch server-side, so staying free
on HuggingFace would mean in-browser inference (ONNX + JavaScript, including a
reimplementation of the CWT and R-peak detection) - a substantial rewrite.

### Day 2: credentials

A HuggingFace token reached `README.md` through a stray right-click paste during
`hf auth login`. Caught before any commit. A local pre-commit hook now blocks
staged changes that add credential-shaped strings (HuggingFace, GitHub, Kaggle
key); it is not versioned, so reinstall it on a fresh clone.
`tests/test_no_secrets.py` scans every tracked file, locally and in CI.

### Day 2: container re-validated on both classes

After the rebuild: normal 5/8, abnormal 7/8, AUC 0.938 over 16 segments, one per
record. The day-1 validation used six abnormal segments only.

## Done since
- DVC initialised and every stage wired; `pipeline_dag.md` regenerated.
- `scripts/remote/` complete: `01_sync_dataset`, `02_make_kernel`,
  `03_run_kernel`, `04_merge_results`, plus two additions beyond the brief —
  `05_run_queue` (Kaggle caps batch GPU sessions at 2, so ~60 jobs need a
  scheduler, and the quota ledger has to come from somewhere) and
  `06_reconcile` (the silent-missing-fold guard above).
- FastAPI service, Gradio demo, CPU-only training and inference Dockerfiles,
  TorchScript export with a trace-vs-eager equivalence check, GitHub Actions CI.
  All written and tested; the Docker images have **not been built** and the
  Spaces deploy is documented rather than attempted (no token provided).

## Datasets on Kaggle
- `ecg-pcg-fusion-scalograms` — default config + manifests +
  negative-control manifests + training scripts. 1.41 GB.
- `ecg-pcg-fusion-ablation` — the six non-default wavelet configs. 8.42 GB,
  uploaded in 18:54. Kept separate so adding it does not force a re-upload of
  the default set, which is why the kernel resolves scalograms and manifests
  from independently-searched mounted datasets.

### Docker images built and smoke-tested — and the build found a real bug

The inference image was written long before it was built. Building it exposed a
defect that no local test could:

`load_model()` did `checkpoint["model"]` unconditionally. That is right for a
training checkpoint (`*_best.pth`, a dict with a "model" key) but wrong for the
**TorchScript export the image actually ships**, which deserialises to a
`RecursiveScriptModule`. Subscripting it raises `NotImplementedError`, and the
container died at startup with exit code 3. The API tests passed throughout,
because they point at the `.pth`.

`load_model()` now detects which form it was handed and behaves accordingly.

**A genuine tension in the brief, resolved explicitly.** §13 says the inference
image should carry "only a TorchScript export", and also that the service must
provide `/explain`. Both cannot hold: Grad-CAM registers backward hooks on
`CNNBranch.features`, and a scripted module's internals do not accept them. So
the image ships **both** artifacts (~5 MB each):

| artifact | purpose |
|---|---|
| `cross_attn_fusion.pt` | TorchScript graph, for graph-only serving |
| `cross_attn_fusion_eager.pth` | state dict — the default, so `/explain` works |

`MODEL_CHECKPOINT` points at the eager form. `/health` reports
`supports_explain`, and `/explain` returns **501 with an explanatory message**
rather than a stack trace if a scripted module is loaded.

**Smoke test against the running container, on real held-out test data:**
- `/health` → `ok`, `model_loaded: true`, `supports_explain: true`,
  `preprocessing: matches params.yaml`
- `/predict` → **6/6 correct** on `cv_fold0` test segments
- `/explain` → HTTP 200, `image/png`, 240 760 bytes, valid PNG

Images: `ecg-pcg-serve:latest` (1.71 GB), `ecg-pcg-train:latest`.
No Docker Hub sign-in is needed — `python:3.11-slim` pulls anonymously.

## Pending
- `warm_start_fusion` needs the `ecg_only` and `pcg_only` checkpoints shipped as
  a third dataset before it can run.
- Wavelet ablation runs, negative-control runs, Grad-CAM, results tables.
- Reconcile the wall-clock ledger rows written before the GPU-time fix.
