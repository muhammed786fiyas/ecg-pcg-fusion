# PROJECT_CONTEXT.md

**ECG–PCG Cardiac Abnormality Detection — dual-branch CWT scalogram fusion with
cross-modal attention.**

| | |
|---|---|
| Local path | `E:\PROJECTS\MAIN PROJECTS\1-CARDIAC-PROJECT-UPDATED` |
| Git remote | `https://github.com/muhammed786fiyas/ecg-pcg-fusion.git`, branch `main` |
| Owner | Muhammed Fiyas |
| Target venue | Biomedical Signal Processing and Control (Elsevier) |
| Dataset | PhysioNet/CinC 2016 Training-A, 405 records with synchronous ECG + PCG |
| Python env | conda env `ecg_pcg`, Python 3.11.16 |
| Compute | Local CPU for the data pipeline; Kaggle free GPU (T4) for training |

---

## Status as of 2026-09-10

Numbering below follows `KICKOFF_PROMPT.md` §15.

**Done — steps 1–7, all committed locally (nothing pushed)**

| § | Work | Outcome |
|---|---|---|
| 1 | Clear the slate | Only `data/raw/physionet2016_training_a/` survives |
| 2 | Skeleton, `params.yaml`, `dvc init`, `.env`, `CLAUDE.md`, this file | Python 3.11.16 recorded; `numpy<2` pinned for `neurokit2==0.2.7` |
| 3 | Data pipeline `01`–`06`, run on the full dataset | 409 → **405 records** → **3752 segments** → **15 008 augmented rows** |
| 4 | Scalograms (default config) + manifests | Two uint8 memmaps, `(15008, 224, 224)`, 718 MB each |
| 5 | Test suite, green *before* any training | **101 tests** passing; ruff clean |
| 6 | Local CPU smoke test | Checkpoint + MLflow parent/child runs + metrics row all produced |
| 7 | Kaggle offload + dry run + merge | Full round trip proven: push → poll → fetch → merge → results table |

**Gate counts (measured, not estimated)** — full table in
`docs/logs/tasks/1-data-pipeline.md`.

| Gate | In | Out | Dropped |
|---|---|---|---|
| `01_convert` | 409 | 405 | 4 (the declared PCG-only records) |
| `02_record_qc` | 405 | 405 | 0 |
| `03_segment` | 405 records | 3752 segments | 0 records |
| `04_segment_qc` | 3752 | 3752 | 0 (194 repaired by interpolation) |
| `05_assign_folds` | 405 | dev 243/60/102; CV 5 × (275/49/81) | — |
| `06_augment` | 3752 | 15 008 rows | — |

Label balance: **117 normal / 288 abnormal** records (71.1% abnormal);
1087 / 2665 at segment level. Abnormal share per CV test fold: 0.716, 0.716,
0.716, 0.704, 0.704.

**Measured throughput.** ~500 s/epoch on this CPU vs **~27 s/epoch on a Kaggle
T4** — an 18× speedup. Full 5-fold CV across all seven families would be ~120 h
locally, so the GPU offload is load-bearing, not an optimisation. Kaggle allows
only **2 concurrent batch GPU sessions**, which is why
`scripts/remote/05_run_queue.py` exists.

**In progress right now**
- **Main training queue running**: 30 jobs (6 families × 5 folds) on the padded
  scalograms. `ecg_only` folds 0–3 complete.
- **Wavelet ablation scalograms regenerating**: w2–w6 written, w7 to go. Every
  config is checked with `scripts/features/verify_pad.py` before it feeds
  training.
- Default scalograms regenerated and verified; manifests rebuilt; Kaggle dataset
  re-synced (v3, padded scalograms + negative-control manifests).

**Verification of the fix — both checks passed**
- Numeric, via `verify_pad.py`: ECG edge/interior energy ratio **19.29 → 0.80**,
  interior mean **4.6 → 60.5** of 255.
- Visual: the ECG scalogram now shows three low-frequency energy concentrations
  (the cardiac cycles) with high-frequency QRS transients beneath, aligned in
  time with the PCG's S1/S2 bursts. No edge bands. The visual check is not
  optional here — the numeric QC gates passed while the images were wrong.

**First complete family — `ecg_only`, 5-fold record-level CV, post-fix**

| fold | segment AUC | patient AUC |
|---|---|---|
| 0 | 0.8406 | 0.8703 |
| 1 | 0.7769 | 0.8231 |
| 2 | 0.7241 | 0.7279 |
| 3 | 0.9446 | 0.9708 |
| 4 | 0.8911 | 0.9276 |
| **mean ± std** | **0.8355 ± 0.0878** | **0.8639 ± 0.0945** |

The spread across folds is large (0.72–0.94 segment AUC). With only 81 test
records per fold that is unsurprising, but it means single-fold numbers are not
meaningful and the ± must always be reported.

**Headline interim finding — concatenation fusion does not beat ECG alone.**
Paired per-fold comparison against `ecg_only` (same folds, same seed), patient
AUC: `pcg_only` is -0.2117, losing all 5 folds (p = 0.001); `dual_cnn` is
-0.0136, losing 4 of 5 folds (p = 0.334). PCG alone is clearly weaker; naive
fusion shows no advantage over ECG alone. The old pipeline reported the opposite
ordering (fusion 0.817 > ECG 0.795) from a patient-level-leaking split, and that
ordering does not reproduce under a clean record-level split. Whether *any*
fusion scheme beats the ECG branch is what the remaining families exist to
answer. Full analysis in `docs/logs/tasks/3-modeling.md`; script is
`scripts/evaluation/05_paired_model_comparison.py`.

**Always run `scripts/remote/06_reconcile.py` before building results tables.**
It merges any kernel output that failed to merge and prints which (family, fold)
pairs are missing. `kaggle kernels output` can return non-zero while delivering
every file — seen on `cross_attn_resnet18`, whose ~280 MB checkpoints are far
larger than the other families' — which previously skipped the merge and left a
completed fold absent from the store with nothing failing loudly.

**A contamination incident, caught and corrected.** A cleanup command chained
with `&&` short-circuited on a busy file, so the rest of the chain never ran and
pre-fix MLflow runs survived the rebuild, sitting alongside their post-fix
replacements. Six stale runs were deleted; the table above is post-fix only.
`03_build_results_tables.py` now hard-fails on duplicate (family, config, fold)
runs, which is the visible symptom of exactly this. See the daily log.

**Next up, in order**
1. ~~Re-sync the Kaggle dataset~~ — done, v3 carries the padded scalograms and
   the negative-control manifests.
2. **Running now:** the 30-job queue (§15.8: `ecg_only`, `pcg_only`, `dual_cnn`,
   `cbam_fusion`, `cross_attn_fusion`, `cross_attn_resnet18`).
3. **Running now:** regenerating the 6 non-default wavelet configs. Every config
   is checked with `scripts/features/verify_pad.py` before it feeds training.
4. `warm_start_fusion` — must run *after* the unimodal folds, since it needs
   their checkpoints shipped as a second Kaggle dataset.
5. Wavelet ablation: sync configs as a second dataset version, 30 runs (§15.9).
6. Split-protocol negative controls, arms B and C (§15.10).
7. Patient aggregation and Grad-CAM on the headline model (§15.11).
8. Results tables and all figures (§15.12).
9. Reconcile the GPU ledger from the fetched kernel logs (see below), then
   finalise this file, tag the milestone, write the summary (§15.14).

**Known limitation to state in the paper — Contribution 1 is confounded.**
Holding the CWT scales fixed across wavelets (as the brief requires) does *not*
hold the frequency band fixed, because each wavelet has its own centre
frequency. At these scales `cmor1.5-1.0` covers 4-100 Hz on ECG while `mexh`
covers 1-25 Hz - a 3.2x spread. So the wavelet comparison confounds wavelet
*shape* with frequency *coverage*. The measured band is printed per row in
`reports/figures/wavelet_table.md` and the caption says so outright. Describe it
as a comparison of wavelets at fixed scales, and name a frequency-matched study
as future work. Full table in `docs/logs/tasks/2-features.md`.

**Compute budget — read the ledger carefully.** The queue that is running
records *wall clock*, which over-counts GPU time by ~4.5× because it includes
Kaggle's scheduling queue. True GPU time per `ecg_only` fold is ~8 min, not
~36 min. `05_run_queue.py` has been fixed to charge GPU seconds read from the
kernel log; the in-flight queue's rows still need reconciling before any total
is quoted. Details in `docs/logs/tasks/5-mlops.md`.

**Pending / not yet started**
- Everything from §15.8 onward. MLOps (§15.13) is written and tested but the
  Docker images have not been built, and the Gradio Spaces deploy is documented
  rather than attempted (no token was provided).

**Known open items**
- `tests/test_api.py` was written after the rest of the suite, once
  `scripts/serving/app.py` existed. It is green.
- The GPU quota ledger (`docs/logs/kaggle_quota_ledger.csv`) was reset along
  with the discarded kernels; ~0.16 GPU-hours were spent on work that was
  thrown away.

---

## Scope

Full pipeline + four contributions + full MLOps stack. Stop before writing the
paper itself, but every figure, table and number the paper needs must exist in
`reports/` at the end.

**The four contributions**

1. **CWT wavelet sensitivity.** Vary each modality's mother wavelet over
   `{cmor1.5-1.0, morl, mexh, gaus4}` holding the other at default; 7 scalogram
   sets + the reused default = 8 table rows. `dual_cnn` architecture throughout,
   so the comparison is about the input representation and nothing else.
2. **Grad-CAM physiological validation.** Grad-CAM on `cross_attn_fusion`,
   hooking the last `Conv2d` of each branch. Does ECG attention land on the QRS
   time–frequency region and PCG attention on the S1/S2 bursts? Report honestly
   either way.
3. **Segment-to-patient aggregation.** Majority vote vs mean probability vs max
   probability, patient-level accuracy / F1 / AUC / sensitivity / specificity.
4. **Split-protocol sensitivity — a deliberate negative control.** `dual_cnn`
   trained three ways: (A) correct record-level, (B) leaky segment-level
   validation with a clean test set, (C) fully leaky segment-level validation
   and test. Quantifies how much apparent performance is manufactured by split
   choice.

---

## The split / evaluation protocol

**Governing rule: the patient is the unit of splitting.** Every split is decided
over record IDs. `scripts/data/05_assign_folds.py` receives only
`(record_id, label)` pairs and a seed — never a segment ID, a signal, or
anything derived from one. `tests/test_split_inputs.py` asserts this structurally.

**Pipeline order.** Segmentation runs *before* fold assignment, on all records at
once. This is not a leak: segmentation is a per-record deterministic operation
(`nk.ecg_peaks` plus fixed non-overlapping 3-second windows, no RNG), so record
`a0014`'s segments are byte-identical whichever fold it later lands in. Running
segmentation first means fold assignment happens over the records that actually
survive QC, so fold sizes and class balance are correct rather than degraded by
post-hoc dropouts.

**Two protocols.**

- **Development** (all exploration, sanity checks): fold 0 only. One stratified
  record-level partition, ~60% train / ~15% inner-val / ~25% test of surviving
  records.
- **Final** (every number in the paper's main table): 5-fold stratified
  record-level cross-validation over all surviving records. Each fold holds out
  ~20% of records as test; within the remaining 80%, ~15% of *records* are held
  out as inner-validation for early stopping. Report mean ± std across folds,
  never the best fold.

`k` comes from `params.yaml` (`evaluation.cv_folds`, default 5).

**Augmentation applies to the training portion only.** Inner-validation and test
are never augmented — validation exists to estimate real-data performance so
early stopping picks the right checkpoint, and augmented copies of the same
segment are near-duplicates that make the metric look more precise than it is.
Enforced at manifest level: val and test manifests carry `_orig` rows only.

**Fold-local statistics.** `pos_weight`, any normalization statistic, and any
decision threshold moved off 0.5 are fitted inside the fold's training portion.
Per-record and per-image normalizations are safe by construction; a global
mean/std is not. This is the second most common leak after the split itself and
it is invisible — the code runs fine and the number is quietly optimistic.

**The whole split is one auditable table**,
`data/processed/manifests/fold_assignments.csv`, with columns `record_id`,
`label`, `dev_partition`, `cv_fold`, and one `cv_inner_val_fold_<i>` boolean
column per fold.

---

## Architecture decisions and their history

**Why a clean-slate rebuild.** The previous pipeline was built in Colab
notebooks. Two defects made it unsalvageable rather than fixable: augmentation
drew from a single unseeded global RNG stream in a loop (so output depended on
filesystem iteration order and could not be reproduced), and the validation
split leaked at the patient level (segments from the same record appeared in
both training and validation). Its reported test AUCs — fusion 0.817,
ECG-only 0.795, PCG-only 0.647 — are therefore **not a benchmark to reach**, and
no number from it is carried forward as a target.

**Models.** Baselines `ecg_only` / `pcg_only` / `dual_cnn` share a custom
4-block CNN branch (channels 1→32→64→128→256, `AdaptiveAvgPool2d` to 256-d; the
feature map before pooling is 14×14×256 at 224×224 input). `cbam_fusion` adds
channel + spatial attention on each branch's 14×14×256 map. `cross_attn_fusion`
(headline) flattens each map to 196 tokens of dim 256 and runs bidirectional
4-head cross-attention — ECG queries PCG, PCG queries ECG — one layer each
direction. `cross_attn_resnet18` swaps in a pretrained ResNet-18 per branch.
`warm_start_fusion` is `dual_cnn`'s architecture with each branch initialised
from that fold's already-trained unimodal checkpoints.

**Why `warm_start_fusion` exists.** A 2026 controlled ablation on this exact
dataset (Kıymık, *Physiological Measurement*) found that dual warm-start
concatenation matched or beat attention and gated-attention fusion, with
attention becoming *inferior* at higher label ratios. If that replicates here it
is a finding worth reporting, not a disappointment. It costs one extra training
run per fold and no extra preprocessing, because the unimodal checkpoints
already exist.

---

## Environment notes

**CPU-only local machine.** AMD Ryzen AI 350 class, no CUDA GPU. All model
training is pushed to Kaggle as batch kernels via the `kaggle` CLI. Training
code stays device-agnostic: AMP only when `torch.cuda.is_available()`, plain
fp32 otherwise, so the identical script runs locally for smoke tests and on a
Kaggle T4 for real.

**Kaggle limits designed around:** 30 GPU hours/week, sessions hard-capped at
9 hours, `machine_shape: NvidiaTeslaT4` (Kaggle's docs warn `NvidiaTeslaP100` is
not usable with the default image due to PyTorch compatibility),
`enable_internet: true` or in-kernel pip installs fail. One kernel per
(model config, fold), so no job approaches the ceiling and a failure costs one
fold rather than a sweep. There is no quota API — usage is tracked by hand in
`docs/logs/tasks/5-mlops.md`.

**`db4` → `gaus4` substitution.** The original plan proposed `db4` as one of the
ablated wavelets. It cannot be used: `pywt.cwt(signal, scales, "db4", ...)`
raises `AttributeError: 'pywt._extensions._pywt.Wavelet' object has no attribute
'complex_cwt'`, because `db4` is a discrete orthogonal wavelet and not a member
of `pywt.wavelist(kind="continuous")`, which is
`['cgau1'..'cgau8', 'cmor', 'fbsp', 'gaus1'..'gaus8', 'mexh', 'morl', 'shan']`.
`gaus4` is a real-valued continuous wavelet in the same spirit as `mexh` and
replaces it. **Flag this in the paper's Method section** — it is a documented
substitution, not a silent swap.

**Record ID vs patient ID — an explicit assumption.** PhysioNet 2016 Training-A
is treated in the literature as one record per subject, and the challenge
documentation distributes it as a flat set of recordings with no subject-ID
field. We therefore treat `record_id` as the patient identifier. **If two
records in Training-A came from the same person, record-level splitting is not
patient-level splitting and this protocol rests on an unverified assumption.**
The dataset ships no information that would let us check, so this is stated as
an assumption rather than a verified fact, and it must be stated the same way in
the paper.

**Scalograms are stored as uint8 memmaps, not PNGs.** One memmap per
(wavelet_config, modality) of shape `(N, 224, 224)`, plus a row-index CSV
mapping `segment_variant_id → row`. Rendering a scalar magnitude field through
matplotlib's `jet` colormap into a 3-channel PNG — what the old pipeline did —
triples the data, discards precision through a perceptually non-uniform colour
mapping, and makes `plt.savefig` plus PNG decode the dominant cost of every
training epoch. A uint8 memmap loads essentially for free and keeps the
scalogram as the scalar field it actually is. `jet` PNGs are rendered separately
and only for the ~30 segments needed for paper figures and Grad-CAM overlays.
Models take 1-channel input; the pretrained ResNet-18 branch replicates the
single channel to 3 at load time.

**The CWT boundary artifact — the most consequential defect found so far.**
The first rendered paper figure showed the ECG scalogram as two bright vertical
bands at the edges with near-black in between. Measured: the outer columns
carried **~19× the interior energy**, and after per-image min/max scaling the
real cardiac content sat at a mean of **4.6/255** with **85% of pixels below
26/255**. PCG was unaffected (ratio ~1.0).

The cause is the CWT cone of influence. A wavelet at scale *s* has effective
support of a few times *s*; the ECG scales run to 500, so at the top of the
range the wavelet is as long as the 6000-sample window itself and the boundary
effect swamps the transform. PCG escapes it because its scales stop at 130.

Fixed by reflect-padding the signal by `4 × max(scale)` before transforming and
cropping the magnitude back to the real window. A measured pad sweep gives
edge/interior 19.29 → 0.63 and interior mean 4.6 → 59.3 by pad = 1000,
converged by 2000 — **about 13× more usable dynamic range**, at ~1.5× the CWT
cost.

This cost 4 completed GPU kernels and 3 scalogram configs, all discarded. That
was still the cheap moment: the bug would not merely have cost accuracy, it
would have produced a **false scientific conclusion**. Grad-CAM would have
highlighted the boundary artifact, and Contribution 2 would have reported "the
model does not attend to the QRS time–frequency region" as a physiological
finding rather than as an artifact of our own preprocessing.

**Every numeric QC gate passed while this was wrong.** The stage checks for
constant images and non-finite values and found nothing. It took rendering one
PNG and looking at it. Full working in `docs/logs/tasks/2-features.md`.

**`numpy<2` is pinned** because `neurokit2==0.2.7` predates the NumPy 2 ABI
break.

**`conda run` crashes when its stdout is piped** and the output is large — an
unhandled conda plugin error. It killed a 1.4 GB Kaggle upload midway, leaving
an empty dataset registered. For long or noisy commands call the interpreter
directly: `C:/Users/muham/.conda/envs/ecg_pcg/python.exe` and
`.../envs/ecg_pcg/Scripts/kaggle.exe`.

**Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<slug>`**, not
`/kaggle/input/<slug>`, and `mlflow` is absent from the default image. Both are
handled in `scripts/remote/02_make_kernel.py`; the rest of the Kaggle gotchas
are in `docs/logs/tasks/5-mlops.md`.

**Grad-CAM is hand-rolled** rather than taking the `grad-cam` package: it is
~40 lines of forward/backward hooks, and keeping it inline preserves the
self-contained-script convention that lets a training or evaluation script run
unmodified inside a Kaggle kernel.

**No shared utils module.** Each script carries its own constants and helpers.
The `Dataset` class and the model definitions are duplicated across the training
scripts on purpose — that duplication is what makes a Kaggle kernel "the same
file with different path constants" rather than a fork.

---

## Literature / citation map

Where each reference is cited in the paper.

**Competing ECG–PCG fusion on PhysioNet 2016** — *Related Work*, and *Results/
Discussion* for comparison: PACFNet (PeerJ CS 2025), TF-CrossNet (BPEE 2025),
CAD-ViT (IEEE JBHI 2025), DDR-Net (BSPC 2024), HS-MMNet (Physiol Meas 2026),
Calzoni et al. (J Med Syst 2025), Bargarai et al. (Diagnostics 2026,
quality-aware fusion + 10-fold CV), Wang et al. (Sensor Review 2025, CWT +
improved ResNet-18 on synchronised PCG–ECG — the closest published analogue to
our ResNet-18 row).

**The paper our ablation must answer** — *Introduction*, *Related Work*,
*Discussion*: Kıymık (Physiol Meas 2026), a controlled warm-start vs attention
ablation on Training-A finding no reliable advantage for attention fusion. Our
`warm_start_fusion` row is the replication.

**Leakage and evaluation protocol** — *Introduction*, *Method* (§4), *Results*
(§10.4): Yoshizawa et al. (IEEE Sensors 2023, quantified segment/record/
subject-level leakage in PPG/ECG blood-pressure estimation), Eltawil et al.
(JCDD 2026, leakage commentary), Ameen et al. (Sci Rep 2026, leakage-safe
recording-level splits on PhysioNet 2016), Singh et al. (Diagnostics 2026,
recording-level stratified 5-fold CV).

**Time–frequency representation** — *Related Work*, *Method*: Singh et al.
(Diagnostics 2026, CWT vs synchrosqueezed CWT on PhysioNet 2016) is the nearest
prior work to Contribution 1. **Our study differs in comparing mother wavelets
per modality inside a bimodal fusion model, rather than transform variants on
PCG alone** — say that explicitly rather than claiming the space is empty.

**Interpretability** — *Method*, *Results*: Grad-CAM (ICCV 2017), Oliveira et al.
(EMBC 2024), Alqudah et al. (Health Inf Sci Syst 2025, Grad-CAM/SHAP/IG on
dual-branch ECG–PCG with cross-modal attention), Suchithra et al. (Array 2026),
Althaph et al. (Sci Rep 2025). Contribution 2 is an **extension** of an active
line, not a first — frame it as validation on dual-CWT scalograms specifically.

**Foundations and context** — *Introduction*, *Method*: PhysioNet/CinC 2016
(Liu et al.), Pan–Tompkins (IEEE TBME 1985), NeuroKit2 (BRM 2021), CBAM
(ECCV 2018), PyWavelets, Zhu et al. (Electronics 2024, review of 104 PhysioNet
2016 papers), CardioState-JEPA (arXiv 2608.12944, 2026, cross-modal cardiac
foundation model — cite as the direction the field is moving, and as future
work).

---

## Folder layout as it actually exists

```
1-CARDIAC-PROJECT-UPDATED/
├── CLAUDE.md                   # conventions + split rules that must not drift
├── PROJECT_CONTEXT.md          # this file
├── KICKOFF_PROMPT.md           # the original brief
├── params.yaml                 # tunable knobs only, nested <module>.<substage>
├── dvc.yaml / dvc.lock / .dvc/
├── pipeline_dag.md             # dvc dag --md, regenerated after adding stages
├── .env                        # gitignored; .env.example is the template
├── configs/
├── scripts/{data,features,modeling,evaluation,serving,remote}/
├── data/
│   ├── raw/physionet2016_training_a/   # never modified by any script
│   ├── interim/{records,segments,augmented}/
│   └── processed/{scalograms/<config>/, manifests/}
├── models/{<family>/, mlflow_tracking/}
├── reports/{<family>/, gradcam/, patient_aggregation/, figures/}
├── tests/
├── docker/
├── .github/workflows/
└── docs/logs/{daily/, tasks/}
```

`data/`, `models/` and `reports/` are DVC-tracked, not git-tracked.
