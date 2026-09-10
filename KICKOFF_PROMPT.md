# ECG–PCG Cardiac Abnormality Detection — Clean-Slate Rebuild Prompt

Paste everything below the line into a fresh Claude Code session opened in
`E:\PROJECTS\MAIN PROJECTS\1-CARDIAC-PROJECT-UPDATED`
(git remote `https://github.com/muhammed786fiyas/ecg-pcg-fusion.git`, branch `main`).

## Do these two things first — Claude Code cannot do them for you

All three are one-time, and training on Kaggle will not start without them.

1. **Confirm the conda environment.** This project uses a dedicated env named `ecg_pcg` — never
   `base`, never the football project's env. Recommended: recreate it on Python 3.11 so it matches
   what Kaggle kernels run.
   ```
   conda deactivate
   conda env remove -n ecg_pcg -y
   conda create -n ecg_pcg python=3.11 -y
   conda activate ecg_pcg
   python -m pip install --upgrade pip
   python -m pip install kaggle
   ```
2. **Phone-verify your Kaggle account.** kaggle.com → your avatar → Settings → Phone Verification.
   Accelerators (GPU/TPU) and notebook internet access are locked until you do this.
3. **Configure Kaggle API credentials.** `pip install --upgrade kaggle` (need >= 1.8.0), then use
   any one of these — the CLI accepts all three:
   - `kaggle auth login` — browser OAuth, no files to place. Simplest.
   - kaggle.com/settings → **API Tokens** → *Generate New Token*, then save the token string to
     `C:\Users\<you>\.kaggle\access_token` (a file with no extension — Notepad will silently
     append `.txt`, so save with quotes around the filename or use VS Code).
   - Same page → **Legacy API Credentials** → *Create Legacy API Key*, which downloads a ready-made
     `kaggle.json` to drop at `C:\Users\<you>\.kaggle\kaggle.json`. Least fiddly on Windows.

   Confirm it works with `kaggle datasets list` before starting. Never commit these credentials.

---

I'm rebuilding this project from scratch. **Everything derived from the raw data gets thrown away
and regenerated properly** — the existing pipeline was built in Colab notebooks with unseeded
augmentation and a validation split that leaked at the patient level, and it isn't worth
salvaging. The only thing that survives is the raw PhysioNet data.

The project: **dual-branch CWT scalogram fusion with cross-modal attention for cardiac abnormality
detection** on PhysioNet/CinC 2016 Training-A (synchronous ECG + PCG, 405 usable records),
targeting a paper in **Biomedical Signal Processing and Control** (Elsevier). Owner: Muhammed Fiyas.

This is a **single autonomous session**. Every decision below is final — do not stop to ask me
anything. If you hit a fork not covered here, take the more reproducible/conservative option,
write down why in `docs/logs/daily/`, and keep going. Commit locally after each numbered
milestone; **never run `git push`** — I review and push myself.

## 0. Non-negotiable decisions

1. **Clean slate, raw data only.** Delete every derived artifact and rebuild from
   `DATASET/1-PHYSIONET RAW DATA/training-a`. I keep a full copy of the current repo in a separate
   local folder, so nothing here needs preserving — delete freely (§1).
2. **Patient-level separation is the top correctness requirement.** Every split — train, val,
   test, every CV fold — is decided over *record IDs*, never over segments or augmented rows. This
   is enforced by code and by an automated test, not by discipline (§12).
3. **Everything is seeded and reproducible.** A single `GLOBAL_SEED` in `params.yaml`; augmentation
   seeded *per segment*, not from a global RNG stream (§5.5). Re-running the pipeline must
   reproduce byte-identical outputs. The old pipeline's augmentation was unseeded — that's a
   large part of why we're starting over.
4. **Split compute: local CPU for the pipeline, Kaggle free GPU for training.** This machine
   has no CUDA GPU (AMD Ryzen AI 350 class), so all model training is pushed to Kaggle as batch
   kernels via the `kaggle` CLI (§9). Training code must still be device-agnostic and must not call
   `torch.amp.autocast(device_type="cuda")` or `GradScaler()` unconditionally — use AMP only when
   `torch.cuda.is_available()`, plain fp32 otherwise, so the identical script runs locally on CPU
   for smoke tests and on a Kaggle T4 for real. Read §9 before launching any training.
5. **Python environment: a dedicated conda env named `ecg_pcg`, which already
   exists.** Never install into `base` and never create another env. Each of your shell calls is a
   fresh shell, so `conda activate` does **not** persist between calls — prefix every
   Python-touching command with `conda run -n ecg_pcg`, for example
   `conda run -n ecg_pcg python scripts/data/01_convert.py --output ...` and
   `conda run -n ecg_pcg dvc repro`. Keep the `cmd` entries in `dvc.yaml` as plain
   `python scripts/...`: they inherit the environment from whatever invokes `dvc`. Install
   dependencies with `conda run -n ecg_pcg python -m pip install -r requirements.txt` — note
   `python -m pip`, not bare `pip`, because the `pip.exe` launcher on this machine fails with
   "Access is denied".

   **First, run `conda run -n ecg_pcg python --version` and record it in
   `docs/logs/tasks/5-mlops.md`.** If it reports 3.9.x, pin `mlflow<3`, `gradio<5` and `numpy<2.1`
   in `requirements.txt` rather than fighting pip's resolver, and note the pins as a documented
   consequence of the Python version. If it reports 3.10+, take current versions of everything.
   Either way, write only code that runs on both the local interpreter and Kaggle's (~3.11): no
   `match` statements, no `X | Y` union syntax. The no-type-hints convention in §2 already keeps you
   clear of most of this.
6. **Scope: full pipeline + all four contributions (§10) + full MLOps stack.** Stop before writing
   the paper itself, but every figure, table and number the paper needs must exist in `reports/`
   when you're done.
7. **Git: local commits and tags only. Never push.**

## 1. Step 0 — clear the slate

I keep a full copy of this repo in a separate local folder, so nothing here needs preserving.
Delete freely, then commit as the first milestone.

1. **Delete**: `CODE/`, `TRAINED_MODELS/`, `PROJECT_LOG.md`, `README.md`, `requirements.txt`,
   `DATASET/2-MATLAB DATA`, `DATASET/3-SPLIT_DATA`, `DATASET/4-SEGMENTED_DATA`,
   `DATASET/5-AUGMENTED_DATA`, `DATASET/5-AUGMENTED_DATA.zip`, `DATASET/6-SCALOGRAMS`.
2. **Keep**: `DATASET/1-PHYSIONET RAW DATA/training-a`, moved to
   `data/raw/physionet2016_training_a/`. No script ever modifies this folder again.
3. Carry no number from the old pipeline forward as a target. Its validation split leaked at the
   patient level, so its reported test AUCs (fusion 0.817 / ECG-only 0.795 / PCG-only 0.647) are
   not a benchmark to reach — see §11.

Raw data facts (verified): 409 records, of which **405 have both ECG and PCG** and 4 are PCG-only
and must be excluded. All sampled at **2000 Hz**. Files are WFDB (`.hea`/`.dat`); ignore the `.wav`
files, they duplicate the PCG channel.

## 2. Coding conventions (every script, no exceptions)

- Flat, beginner-friendly style: constants in CAPS at the top of the file, a single `main()`
  holding overall flow, `argparse` for CLI inputs, `if __name__ == "__main__": main()` at the bottom.
- Helper functions for genuine units of work; list comprehensions for simple transforms; classes
  only where there's real state (a `Dataset`, an `nn.Module`) — not for organization's sake.
- `try/except` only with named exceptions, only where a specific error is genuinely expected. No
  blanket `except:`.
- `print()` progress as simple f-strings — what happened plus the key numbers. No padding or
  alignment format specs, no emojis, no decorative separators; a plain `=== Section ===` header is
  fine.
- Avoid: walrus operator, f-string debugging syntax (`f"{x=}"`), ternary expressions, type hints,
  `requests.Session()` with retry/backoff logic.
- **No shared utils module.** Each script carries its own constants and helpers rather than
  importing from a sibling. You will duplicate the `Dataset` class and the model definitions
  across ~6 training scripts — that's intended. Note the duplication once in
  `docs/logs/tasks/3-modeling.md` so it reads as a decision, not an oversight.
- Never hardcode absolute paths. Everything comes from `argparse` args or `params.yaml`, resolved
  relative to repo root. (The old scripts hardcoded `E:\PROJECTS\CARDIAC-PROJECT-UPDATED\...` and
  broke silently when the folder was renamed — don't repeat that.)
- Script numbering: zero-padded sequential prefix per folder (`01_`, `02_`, ...). One script = one
  pipeline stage = one `dvc.yaml` stage.
- `os.makedirs(os.path.dirname(args.output), exist_ok=True)` before any write.
- Every stage follows the three-step shape: load → QC gate (hard-fail with `raise SystemExit` on
  structural failure, log what failed) → compute/write.

## 3. Folder skeleton

```
1-CARDIAC-PROJECT-UPDATED/
├── params.yaml                 # tunable knobs only, nested <module>.<substage>
├── dvc.yaml / dvc.lock / .dvc/
├── .env                        # MLFLOW_TRACKING_URI etc, gitignored
├── configs/
├── scripts/
│   ├── data/
│   │   ├── 01_convert.py            # WFDB -> per-record .npz, exclude PCG-only
│   │   ├── 02_record_qc.py          # record-level quality gate
│   │   ├── 03_segment.py            # R-peak-centred 3s windows (NeuroKit2)
│   │   ├── 04_segment_qc.py         # segment-level quality gate
│   │   ├── 05_assign_folds.py       # THE SPLIT — record-level, stratified, seeded
│   │   └── 06_augment.py            # seeded per-segment augmentation
│   ├── features/
│   │   └── 01_scalogram.py          # CWT -> uint8 memmap arrays, per wavelet config
│   ├── modeling/
│   │   ├── dataset_prep/01_build_manifests.py
│   │   ├── dual_cnn/  ecg_only/  pcg_only/  cbam_fusion/
│   │   ├── cross_attn_fusion/       # headline model
│   │   ├── cross_attn_resnet18/
│   │   └── wavelet_ablation/
│   ├── evaluation/
│   │   ├── 01_patient_aggregation.py
│   │   ├── 02_gradcam.py
│   │   └── 03_build_results_tables.py
│   ├── serving/  app.py  gradio_demo.py
│   └── remote/                      # Kaggle offload: sync, templating, push/poll/fetch, merge
│       ├── 01_sync_dataset.py  02_make_kernel.py
│       └── 03_run_kernel.py    04_merge_results.py
├── data/
│   ├── raw/physionet2016_training_a/
│   ├── interim/{records,segments,augmented}/
│   └── processed/{scalograms/<wavelet_config>/, manifests/}
├── models/{<family>/, mlflow_tracking/}
├── reports/{<family>/, gradcam/, patient_aggregation/, figures/}
├── tests/
├── docs/
│   └── logs/{daily/DAYn_DD-MM-YYYY.md, tasks/1-data-pipeline.md ... 5-mlops.md}
├── PROJECT_CONTEXT.md
└── pipeline_dag.md              # `dvc dag --md`, regenerate after adding stages
```

`data/`, `models/`, `reports/` are DVC-tracked, not git-tracked — each gets its own `.gitignore`.
Add `kaggle.json`, `.env` and `*.pth` to the root `.gitignore` before the first commit.

## 4. The split design — this is the part that must be right

**The governing rule: the patient is the unit of splitting, and the split function may only ever
see record IDs and record-level labels.** It must never receive a segment ID, a signal, or any
quantity derived from one. Write `05_assign_folds.py` so that this is structurally true — its
inputs are a list of `(record_id, label)` pairs and a seed, and nothing else.

**Pipeline order and why it's safe.** Segmentation runs *before* fold assignment, on all records
at once. That is not a leak, because segmentation is a per-record deterministic operation
(`nk.ecg_peaks` plus fixed non-overlapping 3-second windows, no RNG) — record `a0014`'s segments
are byte-identical no matter which fold it later lands in. Running it first means fold assignment
happens over the records that *actually survive* QC, so fold sizes and class balance are correct
rather than degraded by post-hoc dropouts. Every downstream artifact inherits its record's fold
assignment through the manifests, and the manifests are the only thing training ever reads.

**Protocol.** Two tiers, both patient-level:

- **Development protocol** (all exploration, the wavelet sweep, sanity checks): fold 0 only. One
  stratified record-level partition — roughly 60% train / 15% inner-val / 25% test of surviving
  records. Fast, one training run per config.
- **Final protocol** (every number that goes in the paper's main results table): **5-fold
  stratified record-level cross-validation** over all surviving records. Each fold: ~20% test
  records held out, and within the remaining 80%, ~15% of *records* held out as inner-validation
  for early stopping. Report mean ± standard deviation across folds, never the best fold.

Set `k` from `params.yaml` (`evaluation.cv_folds`, default 5) so §9's calibration can dial it
down if the measured throughput demands it.

**Which sets get augmented.** Training portion only. Inner-validation and test are **never**
augmented — validation exists to estimate real-data performance so early stopping picks the right
checkpoint, and augmented copies of the same segment are near-duplicates that make the metric look
more precise than it is. This is enforced at manifest level (§5.6).

**Fold-local statistics.** Anything fitted must be fitted inside the fold's training portion:
`pos_weight` class weighting, any normalization statistic, any decision threshold if you move it
off 0.5. Per-record and per-image normalizations are safe by construction; a global mean/std is
not. This is the second most common leak after the split itself and it's invisible — the code runs
fine and the number is just quietly optimistic.

**Record ID vs patient ID.** PhysioNet 2016 Training-A is treated in the literature as one record
per subject, but confirm what you can from the dataset's own documentation and state the
assumption explicitly in `PROJECT_CONTEXT.md` and in `docs/logs/tasks/1-data-pipeline.md`. If two
records could come from the same person, record-level splitting is not patient-level splitting and
the whole protocol rests on an unverified assumption — say so honestly rather than quietly
assuming.

## 5. Pipeline stages in detail

**5.1 `01_convert.py`** — WFDB → one `.npz` per record in `data/interim/records/` containing `ecg`
(float32, physical units mV), `pcg` (float32), `fs`. Exclude the 4 PCG-only records, log which.
Normalize PCG per record by its own max absolute value (per-record, so no cross-record leakage).
Write `records_index.csv` (`record_id,label,n_samples,fs`) from the challenge's label file.

**5.2 `02_record_qc.py`** — record-level gate, before any splitting. Drop and log any record whose
PCG or ECG is majority-NaN, is flat/constant, or is too short to yield a single 3-second window.
Doing this at record level (rather than only per segment, as the old pipeline did) avoids records
that silently contribute zero segments and skew the fold balance.

**5.3 `03_segment.py`** — R-peak detection via `nk.ecg_peaks` (Pan–Tompkins), non-overlapping
3-second windows centred on R-peaks, window length `3.0 * fs`. Skip windows that run off either
end. Records where R-peak detection fails or yields nothing are logged and dropped. Output: one
`.npz` per segment in `data/interim/segments/`, named `<record_id>_seg<NNN>`, plus
`segments_index.csv` (`segment_id,record_id,label`).

**5.4 `04_segment_qc.py`** — per-segment gate: drop segments with >90% NaN in either channel, or
with NaN/Inf anywhere after interpolation. Log the affected records and counts. In the old
pipeline this dropped ~173 train and ~79 test segments; expect the same order of magnitude but
report what you actually get.

**5.5 `05_assign_folds.py`** — the split (§4). Inputs: `(record_id, label)` pairs from the
post-QC index, plus `GLOBAL_SEED`. Outputs `data/processed/manifests/fold_assignments.csv` with
columns `record_id, label, dev_partition, cv_fold` where `dev_partition ∈ {train, val, test}` for
the development protocol and `cv_fold ∈ {0..k-1}` gives the CV test fold. Also assign, per CV
fold, which of that fold's training records serve as inner-validation — store as
`cv_inner_val_fold_<i>` boolean columns, so the whole split is one auditable table.

**5.6 `06_augment.py`** — 4× expansion: `_orig` (byte-identical copy), `_noise` (ECG: Gaussian
σ = 1% of ECG std; PCG: Gaussian σ = 0.02), `_scale` (ECG factor ∈ [0.9, 1.1]; PCG ∈ [0.85, 1.15]),
`_combined` (noise + scaling + PCG temporal shift ±50 ms). **Name the fourth variant `_combined`,
not `_mix`** — the old log called it "signal mixing", which reads as cross-sample mixup to a
reviewer and invites a leakage objection that doesn't apply. Each variant derives only from its
own source segment.

Two things that matter here:
- **Seed per segment**, deterministically: derive the RNG seed from `GLOBAL_SEED` and a stable
  hash of the `segment_id` (e.g. `int(hashlib.sha256(f"{GLOBAL_SEED}:{segment_id}".encode())
  .hexdigest()[:8], 16)`). Do **not** draw from one global RNG stream in a loop — that makes the
  output depend on filesystem iteration order and destroys reproducibility.
- **Augment every surviving segment once**, for all records, regardless of fold. Because each
  variant depends only on its own source segment, and because in 5-fold CV every record is in the
  training portion of k−1 folds anyway, generating once and letting the manifests control *usage*
  is both correct and k× cheaper than regenerating per fold. Manifests for val/test partitions
  reference `_orig` rows only.

**5.7 `features/01_scalogram.py`** — CWT via PyWavelets, per wavelet config from `params.yaml`.
Defaults: ECG `cmor1.5-1.0`, scales `np.arange(20, 501)` (≈0.5–40 Hz); PCG `morl`, scales
`np.arange(7, 131)` (≈20–250 Hz). Take `np.abs(coeffs)`, resize to 224×224.

**Store the model input as a single-channel `uint8` memmap array, not a PNG.** One memmap per
`(wavelet_config, modality)` of shape `(N, 224, 224)` plus a row-index CSV mapping
`segment_variant_id → row`. Rationale, and note it in `docs/logs/tasks/2-features.md`: rendering
a scalar magnitude field through matplotlib's `jet` colormap into a 3-channel PNG (what the old
pipeline did) triples the data, discards precision through a perceptually non-uniform colour
mapping, and makes `plt.savefig` plus PNG decode the dominant cost of every training epoch. A
uint8 memmap loads essentially for free and keeps the scalogram as the scalar field it actually
is. Expect roughly 450 MB per modality per config — fine on disk, and it's the single largest
speedup available to a CPU-bound training loop.

Render `jet`-colormap PNGs **separately and only** for the ~30 segments you need for paper figures
and Grad-CAM overlays, into `reports/figures/scalogram_examples/`.

Models take 1-channel input; for the pretrained ResNet-18 branch, replicate the single channel to
3 at load time (standard grayscale→pretrained-RGB practice).

**5.8 `modeling/dataset_prep/01_build_manifests.py`** — joins `fold_assignments.csv` with the
augmented segment index and the scalogram row-index to emit, per protocol and per fold, three CSVs
(`train/val/test`) listing `row_index, segment_variant_id, record_id, label`. Training manifests
carry all four variants; val and test manifests carry `_orig` only. **These manifests are the only
thing any training script reads** — no training script is allowed to glob a directory.

## 6. DVC

`dvc init`. Stages in order: `convert` → `record_qc` → `segment` → `segment_qc` → `assign_folds` →
`augment` → `scalogram_default` → `scalogram_wavelet_ablation` → `build_manifests` → one `train`
stage per model family → `wavelet_ablation_train` → `patient_aggregation` → `gradcam` →
`build_results_tables`.

`dvc.yaml` owns every file path (`cmd`/`deps`/`params`/`outs`). `params.yaml` holds only tunable
knobs, nested `<module>.<substage>` (e.g. `features.scalogram.ecg_wavelet`,
`modeling.cross_attn_fusion.lr`, `evaluation.cv_folds`) — never paths, so changing one param
doesn't dirty unrelated stages. Regenerate `pipeline_dag.md` via `dvc dag --md` after adding
stages.

## 7. MLflow

File-based: `MLFLOW_TRACKING_URI=file:./models/mlflow_tracking` in `.env`, `load_dotenv()` before
any `set_experiment()`/`start_run()`, and
`os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` inside the scripts themselves (MLflow 3.x
refuses the plain filesystem backend otherwise — don't leave this for me to set by hand).

One experiment, `ecg-pcg-fusion`. For CV runs use **nested runs**: a parent run per configuration
logging the mean ± std across folds, with one child run per fold. Params to log on every run:
`model_family`, `protocol` (`dev` | `cv`), `fold`, `ecg_wavelet`, `pcg_wavelet`, `backbone`,
`fusion`, `lr`, `batch_size`, `epochs_run`, `seed`. Metrics: segment-level accuracy/F1/AUC,
patient-level accuracy/F1/AUC/sensitivity/specificity, per-class precision and recall. The paper's
results table should be readable straight off the MLflow compare view. Register the best
cross-attention model as `ecg-pcg-detector/latest`.

## 8. Models

**Baselines** (the ablation floor):
- `ConvBlock`: `Conv2d(in, out, 3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)`.
- `CNNBranch`: 4 × `ConvBlock`, channels `1→32→64→128→256`, then `AdaptiveAvgPool2d((1,1))` →
  256-d. With 224×224 input the feature map before pooling is **14×14×256**.
- `dual_cnn`: two branches, concat → 512-d → `Linear(512,128) → ReLU → Dropout(0.5) → Linear(128,1)`.
- `ecg_only` / `pcg_only`: one branch, `Linear(256,128) → ReLU → Dropout(0.5) → Linear(128,1)`.
- Loss `BCEWithLogitsLoss(pos_weight=...)` with `pos_weight` from the fold's training manifest.
  Adam, lr 1e-4. Labels mapped `{-1,+1} → {0,1}`.

**`cbam_fusion`**: CBAM (channel attention via shared MLP over avg-pool + max-pool, then spatial
attention via 7×7 conv over channel-wise avg+max maps) applied to each branch's 14×14×256 map
before global pooling; fuse by concatenation as in `dual_cnn`.

**`cross_attn_fusion`** (headline): flatten each branch's 14×14×256 map to 196 tokens of dim 256.
Bidirectional multi-head cross-attention, 4 heads × 64 dim — ECG tokens query PCG keys/values and
vice versa. One layer each direction is enough; don't build a deep transformer here. Mean-pool
each attended sequence back to 256-d, concatenate to 512-d, same classifier head.

**`cross_attn_resnet18`**: `torchvision.models.resnet18(weights="DEFAULT")` per branch with the
final `fc` removed, 1-channel input replicated to 3, 512-d `avgpool` output projected to 256 by a
`Linear`, then the same cross-attention fusion block.

**`warm_start_fusion`**: identical architecture to `dual_cnn` (plain concatenation, no attention),
but each branch is initialised from the already-trained `ecg_only` and `pcg_only` checkpoints for
the same fold rather than from scratch, then fine-tuned end-to-end. This costs one extra training
run per fold and no extra preprocessing, because the unimodal checkpoints already exist. It
matters because a 2026 controlled ablation on this exact dataset (Kıymık, *Physiological
Measurement*) found that dual warm-start concatenation matched or beat attention and
gated-attention fusion, with attention becoming *inferior* at higher label ratios. If that
replicates here, it is a finding worth reporting, not a disappointment — train it and report it
whichever way it lands.

**Training loop**: up to `MAX_EPOCHS_MAIN` epochs, early stopping on inner-validation AUC with
patience 5. Every training script must be **resume-safe** — re-running picks up from the last
checkpoint and skips already-completed MLflow runs rather than restarting. With dozens of CPU runs
queued, assume the session will be interrupted at least once.

## 9. Compute: local CPU for the pipeline, Kaggle GPU for training

The data pipeline (convert → … → scalograms → manifests) runs locally on CPU. Training runs on
**Kaggle's free GPU** as batch kernels driven by the `kaggle` CLI (`pip install kaggle`). There is
no interactive GPU shell — the loop is push, poll, fetch.

**Verified limits; design around them.** 30 GPU hours per week, resetting weekly. Sessions are hard
-capped at **9 hours**. Accelerators are P100 or dual-T4 — set `"machine_shape": "NvidiaTeslaT4"`,
because Kaggle's own docs warn that `NvidiaTeslaP100` is not usable with the default Kaggle image
due to PyTorch compatibility. Set `"enable_internet": "true"` or pip installs inside the kernel
fail.

**What this changes.** On GPU, full 5-fold CV is affordable across every configuration — so run it.
Do not fall back to single-split protocols unless the quota actually runs out.

**`scripts/remote/`:**
- `01_sync_dataset.py` — package `data/processed/scalograms/<config>/`, the manifests, and the
  training scripts into a **private** Kaggle Dataset: `kaggle datasets create` the first time,
  `kaggle datasets version` thereafter. Upload the `default` wavelet config first (~900 MB across
  both modalities); add the six ablation configs as a later version rather than blocking the main
  runs behind a ~6 GB upload.
- `02_make_kernel.py` — emit `kernel-metadata.json` (`kernel_type: "script"`, `language: "python"`,
  `enable_gpu: "true"`, `enable_internet: "true"`, `machine_shape: "NvidiaTeslaT4"`,
  `dataset_sources: ["<user>/<slug>"]`) plus the entry script for one (config, fold) job.
- `03_run_kernel.py` — `kaggle kernels push`, poll `kaggle kernels status` on a sane interval, then
  `kaggle kernels output` into `models/` and `reports/` once complete.
- `04_merge_results.py` — merge returned MLflow run directories into the local file store so §11's
  table builder treats local and remote runs identically.

**The no-shared-utils convention pays off here.** Each training script is already self-contained, so
a Kaggle kernel is that same script with its path constants pointed at `/kaggle/input/<slug>/` and
`/kaggle/working/`. Drive those from argparse defaults or an environment variable so one file runs
in both places. Do not fork the training code into a local copy and a Kaggle copy.

**Job sizing.** One kernel per (model config, fold), so no job approaches the 9-hour ceiling and a
failure costs one fold rather than a sweep. Resume-safety (§8) is load-bearing here, not
aspirational.

**Quota accounting.** There is no quota API, so track it yourself: log each kernel's wall-clock
duration to `docs/logs/tasks/5-mlops.md` and keep a running weekly total. On reaching 25 of the 30
hours, stop launching jobs, record what remains under "next up" in `PROJECT_CONTEXT.md`, and finish
the local work (tables, figures, MLOps) rather than stalling.

**Prove the pipeline locally before pushing anything.** Run one config end-to-end on CPU with
`smoke.enabled: true` in `params.yaml` — a dozen records, 1 fold, 2 epochs — and confirm it produces
a checkpoint, an MLflow run and a metrics row. Only then start pushing kernels. Debugging a broken
pipeline through 40-minute remote round trips is miserable and entirely avoidable.

**Distinguish auth failures from quota failures.** Credentials are configured via
`kaggle auth login` and verified working. If Kaggle calls start failing with an **authentication**
error, the OAuth token has most likely expired — do not fall back to CPU and quietly burn hours on
a one-minute fix. Stop launching kernels, write the problem at the top of `PROJECT_CONTEXT.md`,
carry on with whatever local work remains (tables, figures, tests, MLOps), and say clearly in your
final summary that `kaggle auth login` needs re-running.

**Fallback.** If no Kaggle credentials are configured at all (`~/.kaggle/access_token`,
`~/.kaggle/kaggle.json`, `KAGGLE_API_TOKEN`, or a prior `kaggle auth login` — verify with
`kaggle datasets list`), the weekly quota is exhausted, or the API fails twice consecutively for
non-auth reasons, fall back to local CPU training and degrade in this order, logging which rungs
you used: wavelet ablation to `MAX_EPOCHS_ABLATION = 20` and development protocol only; CV kept only
for `dual_cnn`, `cross_attn_fusion` and one unimodal baseline; `evaluation.cv_folds` 5 → 3;
augmentation 4× → 2× (`_orig` + `_combined`); scalogram resolution 224 → 128 for custom-CNN configs
(ResNet-18 keeps 224). Stop descending as soon as the projection fits `COMPUTE_BUDGET_HOURS`,
default 24. Never degrade by cutting early-stopping patience or shrinking the test set.

Defaults: `MAX_EPOCHS_MAIN = 50`, `MAX_EPOCHS_ABLATION = 25`, `EARLY_STOPPING_PATIENCE = 5`,
`batch_size = 32`. Locally, `torch.set_num_threads(os.cpu_count())` and size `num_workers` from
`os.cpu_count()` rather than a hardcoded 2.

## 10. The novel contributions

**10.1 CWT wavelet sensitivity (Contribution 1).** Vary each modality's wavelet while holding the
other at its default, over `{cmor1.5-1.0, morl, mexh, gaus4}`. Keep the scale ranges fixed across
wavelets — retuning scales per wavelet is a different and much larger experiment.

| # | ECG wavelet | PCG wavelet | Note |
|---|---|---|---|
| 1 | cmor1.5-1.0 | morl | default config, = the `dual_cnn` result |
| 2 | morl | morl | |
| 3 | mexh | morl | |
| 4 | gaus4 | morl | |
| 5 | cmor1.5-1.0 | cmor1.5-1.0 | |
| 6 | cmor1.5-1.0 | mexh | |
| 7 | cmor1.5-1.0 | gaus4 | |

Seven scalogram sets, seven training runs plus the reused default = 8 table rows. Architecture is
`dual_cnn` throughout — this contribution is about the input representation, so don't confound it
with the attention ablations.

**`gaus4` replaces `db4`, which the master plan proposed.** I verified this directly:
`pywt.cwt(signal, scales, "db4", ...)` raises
`AttributeError: 'pywt._extensions._pywt.Wavelet' object has no attribute 'complex_cwt'`, because
`db4` is a discrete orthogonal wavelet and not a member of `pywt.wavelist(kind="continuous")`
(`['cgau1'..'cgau8','cmor','fbsp','gaus1'..'gaus8','mexh','morl','shan']`). `gaus4` is a
real-valued continuous wavelet in the same spirit as `mexh`. Document the substitution explicitly
in `docs/logs/tasks/2-features.md` and flag it for the paper's Method section — don't swap it
silently, and don't try to force `db4` through `pywt.cwt`.

**10.2 Grad-CAM physiological validation (Contribution 2).** Grad-CAM on the trained
`cross_attn_fusion` model, hooking the last `Conv2d` of each branch's `CNNBranch.features`.
Overlay heatmaps on the `jet` PNG renders (§5.7) for 10–15 representative test segments — correctly
classified normal, correctly classified abnormal, and any interesting misclassifications — into
`reports/gradcam/`. Then write an honest interpretation in
`docs/logs/tasks/4-interpretability.md`: does the ECG map concentrate on the QRS complex's
time–frequency region and the PCG map on the S1/S2 bursts? Report it plainly if it doesn't —
a negative interpretability result is still a result, and inventing agreement is worse than not
having it.

**10.3 Segment-to-patient aggregation (Contribution 3).** Group test predictions by `record_id`
and compare three strategies — majority vote, mean probability, max probability — reporting
patient-level accuracy, F1, AUC, sensitivity and specificity alongside the segment-level numbers.
Under the CV protocol, aggregate within each fold's test partition and then report mean ± std
across folds. Run this for `cross_attn_fusion` and `dual_cnn` at minimum.

**10.4 Split-protocol sensitivity — a deliberate negative control.** Train `dual_cnn` three times
with everything held constant except how the data is partitioned:

- **A — correct**: record-level train/val/test, as specified in §4.
- **B — leaky validation**: record-level test kept clean, but the validation set drawn by a random
  split over *segment rows* from the training pool. This is what the old pipeline did and what a
  lot of published work does implicitly; it corrupts early stopping without touching the test set.
- **C — fully leaky**: random segment-level split for validation *and* test.

Report segment- and patient-level AUC for all three and the deltas between them. This quantifies,
on this dataset, how much apparent performance is manufactured by split choice — the effect has
been measured for PPG/ECG blood-pressure estimation (Yoshizawa et al., *IEEE Sensors Journal*
2023) but not, as far as I can find, for ECG–PCG fusion on PhysioNet 2016, where reported
accuracies of 93–99% are common and leakage-controlled protocols have only recently become
standard practice. Two extra training runs, and it is likely the most defensible thing in the
paper.

Label rows B and C as **negative controls** everywhere they appear — in the CSV, in the figure
captions, in MLflow tags. They must never be readable as a result of the method.

## 11. Results tables

`evaluation/03_build_results_tables.py` reads the MLflow experiment and renders
`reports/figures/ablation_table.md` + `.csv`:

| Model | Fusion | Backbone | Seg AUC | Patient AUC |
|---|---|---|---|---|
| ECG-only CNN | — | Custom 4-block | | |
| PCG-only CNN | — | Custom 4-block | | |
| Dual-branch concat | Concatenation | Custom 4-block | | |
| Dual-branch + warm-start | Concatenation | Custom 4-block, unimodal init | | |
| Dual-branch + CBAM | Channel+spatial attn | Custom 4-block | | |
| **Dual-branch + cross-attn** | Cross-modal attention | Custom 4-block | | |
| ResNet-18 + cross-attn | Cross-modal attention | Pretrained ResNet-18 | | |
| Wavelet ablation ×8 | Concatenation | Custom 4-block | | — |

Plus a separate, clearly separated split-protocol table from §10.4 (A / B / C), captioned as a
negative control.

CV rows carry mean ± std. Also generate: ROC curves, training curves, the Grad-CAM panel, the
wavelet-sensitivity table, and a pipeline/architecture diagram, all into `reports/figures/`.

Report whatever the numbers actually are. Do not tune toward the old pipeline's results (§1.3) —
it had a leaking validation split, so its numbers are not a target, and a lower honest number here
is a better result than a higher dishonest one. Published PhysioNet 2016 accuracies in the 93–99%
range mostly predate leakage-controlled protocols; §10.4 exists precisely to quantify that gap
rather than compete with it.

## 12. Tests (`tests/`, pytest)

The leakage guards are the point of this suite — split bugs come back during refactors:
- **`test_leakage.py`**: for the development partition and for every CV fold, assert no
  `record_id` appears in more than one of train/val/test; assert no base segment appears in two
  partitions; assert val and test manifests contain `_orig` rows only.
- **`test_split_inputs.py`**: assert `assign_folds` accepts only record IDs and labels — that it
  cannot see segment-derived data by construction.
- **`test_reproducibility.py`**: augmenting a fixed segment twice with the same seed produces
  byte-identical output.
- **`test_dataset.py`**: tensor shapes, label range `{0,1}`, no NaN in a batch.
- **`test_model.py`**: forward-pass output shapes for all six architectures on synthetic tensors.
- **`test_api.py`**: FastAPI `/predict` request/response schema contract.

## 13. MLOps

**Docker** — training image on a CPU-only PyTorch base (no CUDA base image, this machine has no
GPU and the image shouldn't assume one) plus neurokit2 and PyWavelets; inference image carrying
only a TorchScript export of `cross_attn_fusion` and torch CPU.

**FastAPI** (`scripts/serving/app.py`) — `/health` (model loaded + version), `/predict` (raw 3 s
ECG+PCG at 2000 Hz → class + confidence), `/explain` (same input → Grad-CAM PNG over both
scalograms).

**GitHub Actions** (`.github/workflows/ci.yml`) — `pytest` plus a linter on every push, CPU-only
with synthetic tensors so no data is needed in CI. Badge in `README.md`. It won't trigger until I
push, which is fine.

**Gradio** (`scripts/serving/gradio_demo.py`) — upload ECG+PCG → prediction + confidence +
Grad-CAM. Get it running locally and document the HuggingFace Spaces deploy command in
`docs/logs/tasks/5-mlops.md`, but don't attempt the actual deployment — I haven't given you a
token.

**`requirements.txt`** — rewrite it: numpy, scipy, pandas, scikit-learn, wfdb, `neurokit2==0.2.7`,
PyWavelets, torch, torchvision, matplotlib, Pillow, dvc, mlflow, python-dotenv, pytest, fastapi,
`uvicorn[standard]`, gradio, and a Grad-CAM implementation (`grad-cam` package or hand-rolled —
your call, document which).

## 14. Documentation

**`PROJECT_CONTEXT.md`** (create in Step 0, keep current throughout, not just at the end): local
path, scope, the split/evaluation protocol from §4 stated precisely, architecture decisions and
their history, the folder layout as it actually exists, a `Status as of <date>` section
(done / next up / pending), and an environment-notes section covering CPU-only training, the
`db4→gaus4` substitution, the record-vs-patient-ID assumption, the memmap-not-PNG decision, and
anything awkward to install on this machine.

Also carry a literature/citation map, noting where each paper is cited (Introduction, Related
Work, Method, Results/Discussion):

- *Competing ECG–PCG fusion on PhysioNet 2016* — PACFNet (PeerJ CS 2025), TF-CrossNet (BPEE 2025),
  CAD-ViT (IEEE JBHI 2025), DDR-Net (BSPC 2024), HS-MMNet (Physiol Meas 2026), Calzoni et al.
  (J Med Syst 2025), Bargarai et al. (Diagnostics 2026, quality-aware fusion + 10-fold CV),
  Wang et al. (Sensor Review 2025, CWT + improved ResNet-18 on synchronised PCG–ECG — closest
  published analogue to our ResNet-18 row).
- *The paper our ablation must answer* — Kıymık (Physiol Meas 2026): controlled warm-start vs
  attention ablation on Training-A, finding no reliable advantage for attention fusion. Cite in
  Introduction, Related Work and Discussion; §8's `warm_start_fusion` row is our replication.
- *Leakage and evaluation protocol* — Yoshizawa et al. (IEEE Sensors 2023, quantified
  segment/record/subject-level leakage), Eltawil et al. (JCDD 2026, leakage commentary), Ameen
  et al. (Sci Rep 2026, leakage-safe recording-level splits on PhysioNet 2016), Singh et al.
  (Diagnostics 2026, recording-level stratified 5-fold CV). These justify §4 and §10.4.
- *Time–frequency representation* — Singh et al. (Diagnostics 2026, CWT vs synchrosqueezed CWT
  systematic comparison on PhysioNet 2016) is the nearest prior work to Contribution 1; our study
  differs in comparing *mother wavelets per modality* in a bimodal fusion model rather than
  transform variants on PCG alone. Say that explicitly rather than claiming the space is empty.
- *Interpretability* — Grad-CAM (ICCV 2017), Oliveira et al. (EMBC 2024), Alqudah et al. (Health
  Inf Sci Syst 2025, Grad-CAM/SHAP/IG on dual-branch ECG–PCG with cross-modal attention),
  Suchithra et al. (Array 2026), Althaph et al. (Sci Rep 2025). Contribution 2 is now an
  *extension* of an active line, not a first — frame it as validation on dual-CWT scalograms
  specifically.
- *Foundations and context* — PhysioNet/CinC 2016 (Liu et al.), Pan–Tompkins (IEEE TBME 1985),
  NeuroKit2 (BRM 2021), CBAM (ECCV 2018), PyWavelets, Zhu et al. (Electronics 2024, review of 104
  PhysioNet 2016 papers), CardioState-JEPA (arXiv 2608.12944, 2026, cross-modal cardiac foundation
  model — cite as the direction the field is moving and as future work).

**`docs/logs/daily/DAYn_DD-MM-YYYY.md`** per work session: Work done / Key decisions /
Discussed-not-decided / Blockers / Next.
**`docs/logs/tasks/n-Module.md`** cumulative per module (1-data-pipeline, 2-features, 3-modeling,
4-interpretability, 5-mlops): Scope / Completed / Key decisions / Data notes & gotchas / Pending /
Ideas. Be honest about what's solid versus rough — this is what a future session reads first.

## 15. Order of work

1. Step 0 clear slate (§1). Commit.
2. Skeleton, `params.yaml`, `dvc.yaml`, MLflow, `.env`, `requirements.txt`, `PROJECT_CONTEXT.md`,
   and a root `CLAUDE.md` holding §2's coding conventions, the split rules from §4, and the
   `conda run -n ecg_pcg` rule from §0.5 — this run is long enough that your context will compact,
   and `CLAUDE.md` is what survives that. Commit.
3. Data pipeline `01`–`06` (§5.1–6.6). Run it. Log the real record/segment counts at every gate.
   Commit.
4. Scalograms for the default config (§5.7) + manifests (§5.8). Commit.
5. Write `tests/` (§12) and get them green **before** training anything — the leakage guards are
   worth more before the runs than after. Commit.
6. Local CPU smoke test: one config end-to-end with `smoke.enabled: true` (§9). Confirm checkpoint
   + MLflow run + metrics row. Commit.
7. Build `scripts/remote/` (§9), sync the `default` scalogram config to a private Kaggle Dataset,
   and push **one** kernel as a dry run. Confirm output comes back and merges into the local MLflow
   store before queueing anything else. Commit.
8. Train the baselines (`ecg_only`, `pcg_only`, `dual_cnn`), then `warm_start_fusion` (reuses the
   two unimodal checkpoints), `cbam_fusion`, `cross_attn_fusion`, `cross_attn_resnet18` — 5-fold CV
   each, one kernel per (config, fold). Commit after each config completes.
9. Wavelet ablation: 7 scalogram sets, synced as a second Dataset version, 7 runs (§10.1). Commit.
10. Split-protocol negative control: 2 extra `dual_cnn` runs (§10.4). Commit.
11. Patient aggregation (§10.3) and Grad-CAM (§10.2) on the headline model. Commit.
12. Results tables + all figures (§11). Commit.
13. MLOps: Docker, FastAPI, CI, Gradio (§13). Commit.
14. Finalize `PROJECT_CONTEXT.md`, tag the milestone, and give me a written summary: the real
    pipeline numbers at each gate, the results table with CV mean ± std, which degradation rungs
    §9 forced and what they cost, what the Grad-CAM actually showed, and every point where you
    deviated from this prompt and why.

Go.
