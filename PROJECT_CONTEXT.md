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

## Status as of 2026-09-10 (end of day 1)

Numbering follows `KICKOFF_PROMPT.md` §15.

### Done — steps 1 through 13

| § | Work | Outcome |
|---|---|---|
| 1 | Clear the slate | only raw Training-A survives |
| 2 | Skeleton, params, DVC, CLAUDE.md | Python 3.11.16, `numpy<2` pinned |
| 3 | Data pipeline `01`–`06` | 409 → **405 records** → **3752 segments** → **15 008 rows** |
| 4 | Scalograms + manifests | uint8 memmaps, 7 wavelet configs |
| 5 | Test suite | **101 tests**, green before training, ruff clean |
| 6 | CPU smoke test | checkpoint + MLflow run + metrics |
| 7 | Kaggle offload | push → poll → fetch → merge, proven |
| 8 | 7 families × 5-fold CV | **35/35** |
| 9 | Wavelet ablation | **35/35** (7 configs × 5 folds) |
| 10 | Split-protocol negative controls | **2/2** |
| 11 | Patient aggregation + Grad-CAM | done |
| 12 | Results tables + figures | done |
| 13 | MLOps | FastAPI, both Docker images built and smoke-tested, CI green |

**Compute: 70 kernels, 9.11 true GPU-hours of the 30/week budget.** No
degradation rung from §9 was needed. The ledger records GPU time and wall clock
separately; 29 early rows were reconciled from the kernel logs.

### Main results — record-level 5-fold CV, mean ± std

| family | segment AUC | patient AUC |
|---|---|---|
| `ecg_only` | 0.8355 ± 0.0786 | 0.8639 ± 0.0845 |
| `pcg_only` | 0.6237 ± 0.0753 | 0.6523 ± 0.0919 |
| `dual_cnn` | 0.8353 ± 0.0723 | 0.8504 ± 0.0852 |
| `warm_start_fusion` | 0.8402 ± 0.0486 | 0.8650 ± 0.0569 |
| `cbam_fusion` | 0.8747 ± 0.0554 | 0.8945 ± 0.0685 |
| `cross_attn_fusion` | 0.8218 ± 0.0657 | 0.8508 ± 0.0810 |
| **`cross_attn_resnet18`** | **0.9216 ± 0.0302** | **0.9380 ± 0.0386** |

**Headline finding: no custom-CNN fusion variant is distinguishable from ECG
alone.** Paired per-fold against `ecg_only` (patient AUC): `dual_cnn` −0.014
(p=0.33), `cross_attn_fusion` −0.013 (p=0.35), `cbam_fusion` +0.031 (p=0.17),
`warm_start_fusion` ≈ 0. Only `cross_attn_resnet18` rises above, +0.074
(p=0.095) — and since it shares the identical cross-attention block, **the gain
comes from the pretrained backbone, not the fusion**. `pcg_only` is clearly
worse (−0.212, p=0.001). This replicates Kıymık (Physiol Meas 2026) and
contradicts the old pipeline's fusion-wins ordering, which came from a
patient-level-leaking split.

### Day 2 (2026-09-11) — in progress right now

- **Fixed a train/serve preprocessing skew.** The serving copies of the scalogram
  transform (`scripts/serving/app.py`, `gradio_demo.py`) never received day 1's
  reflect-padding fix; the served model was fed ECG scalograms differing from the
  training memmap by ~59/255 per pixel. Fixed, and guarded permanently by
  `tests/test_serving_parity.py`. **Day 1's "container validated 6/6" claim was
  inadequate** — all six test segments were strongly abnormal.
- **HuggingFace Space for `cross_attn_resnet18` staged and tested locally**
  (`scripts/serving/hf_space_app.py`, `scripts/serving/build_hf_space.py`). Fold 0
  weights, selected by inner-validation AUC 0.9745. Record-level inference over
  R-peak-centred windows with mean aggregation. The bundled normal example
  (`a0038`) is misclassified (p = 0.584) and kept, with ground truth displayed.

- **Docker inference image rebuilt with the fix** and re-tested on both
  classes: normal 5/8, abnormal 7/8, AUC 0.938 over 16 segments (one per record,
  `cv_fold0` test). It still serves `cross_attn_fusion`.
- **HuggingFace refused the upload with HTTP 402**: Gradio and Docker Spaces now
  require a PRO subscription, even on free `cpu-basic` hardware. Nothing was
  created. The staged Space is ready to upload.
- **A HuggingFace token was accidentally pasted into `README.md`**, caught before
  any commit. A local pre-commit hook and `tests/test_no_secrets.py` now block
  credential-shaped strings. The owner should revoke that token.

### Next, in order

1. **Owner decision: where to host the public demo** — HuggingFace PRO, a free
   alternative, or no hosted demo. With PRO it is one command:
   `python scripts/serving/build_hf_space.py --upload`.
2. Revoke the exposed HuggingFace token; create a new one.
3. Grad-CAM on `cross_attn_resnet18` across folds, reported in Hz.

### Known open items

- **`/explain` needs the eager checkpoint.** The Docker image ships both a
  TorchScript graph and a state dict; `MODEL_CHECKPOINT` points at the latter so
  all three endpoints work. §13 asked for "only a TorchScript export", which is
  incompatible with serving Grad-CAM — documented in `docs/logs/tasks/5-mlops.md`.
- **Contribution 1 is confounded** — fixed scales do not mean fixed frequency
  bands. See below.
- **Grad-CAM gives a mixed answer** — PCG matches physiology, ECG does not. See
  `docs/logs/tasks/4-interpretability.md`.
- The public Space is staged but not hosted: HuggingFace now charges for Gradio Spaces.

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
