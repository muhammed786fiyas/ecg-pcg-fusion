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

## Data notes & gotchas
- `pip.exe` fails on this machine with "Access is denied" — always use
  `python -m pip`.
- `conda activate` does not persist between tool calls; every Python command is
  prefixed `conda run -n ecg_pcg`.

## Kaggle GPU quota ledger
Weekly budget 30 h; stop launching new jobs at 25 h (`compute.kaggle_stop_at_hours`).

| Date | Kernel slug | Config | Fold | Wall-clock | Running weekly total |
|---|---|---|---|---|---|
| _(none yet)_ | | | | | 0.00 h |

## Pending
- DVC init and stage wiring.
- `scripts/remote/` (sync, kernel templating, push/poll/fetch, merge).
- Docker images, FastAPI, CI workflow, Gradio demo.

## Ideas
- If the ablation upload (~6 GB) is slow, sync it as a second dataset *version*
  rather than blocking the main runs behind it — already the plan in §9.
