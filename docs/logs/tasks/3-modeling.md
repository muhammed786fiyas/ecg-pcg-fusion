# Task log 3 — Modeling

## Scope
Seven architectures (`ecg_only`, `pcg_only`, `dual_cnn`, `warm_start_fusion`,
`cbam_fusion`, `cross_attn_fusion`, `cross_attn_resnet18`), the training loop,
and the split-protocol negative control.

## Completed
- `ecg_only`, `pcg_only`, `dual_cnn` complete at 5-fold record-level CV on the
  padded scalograms. See "Interim result" below.

## Interim result — concatenation fusion does not beat ECG alone

Patient-level AUC, 5-fold record-level CV, mean ± std across folds:

| family | segment AUC | patient AUC |
|---|---|---|
| `ecg_only` | 0.8355 ± 0.0878 | **0.8639 ± 0.0945** |
| `pcg_only` | 0.6237 ± 0.0841 | 0.6523 ± 0.1027 |
| `dual_cnn` | 0.8353 ± 0.0809 | 0.8504 ± 0.0953 |

Those error bars overlap heavily, which is why the comparison is done **paired**:
every family sees the same folds, the same data and the same seed, so the fold
is a matched unit.

Paired against `ecg_only`, patient AUC:

| comparison | mean diff | folds won | paired t p |
|---|---|---|---|
| `pcg_only` − `ecg_only` | **−0.2117** | 0 of 5 | 0.001 |
| `dual_cnn` − `ecg_only` | −0.0136 | 1 of 5 | 0.334 |

**PCG alone is clearly and significantly worse than ECG alone.** That part is
unambiguous.

**Naive concatenation fusion shows no advantage over ECG alone** — it is very
slightly behind on average and loses on 4 of 5 folds. With k = 5 the test has
almost no power, so the honest statement is *this experiment cannot distinguish
them*, not *they are identical*. But there is certainly no evidence here that
concatenating a much weaker PCG branch onto the ECG branch helps.

**Why this matters to the paper.** The old pipeline reported fusion 0.817 >
ECG-only 0.795, i.e. fusion helping — from a validation split that leaked at the
patient level. Under a clean record-level split that ordering does not reproduce.
It also sets up the question the remaining families exist to answer: does *any*
fusion scheme (CBAM, cross-attention, warm start, pretrained ResNet-18) beat the
ECG branch on its own? If none does, that is the finding, and it lines up with
Kıymık (Physiol Meas 2026) reporting no reliable advantage for attention fusion
on this same dataset.

Reported by `scripts/evaluation/05_paired_model_comparison.py`.

## Does the fusion model use the PCG? Modality permutation test

`scripts/evaluation/07_modality_permutation.py`. Within each fold's test
records, one modality is replaced by the same modality from a different record
(a record-level derangement, 5 permutations per fold, averaged), and the
record-level AUC is recomputed. Nothing is fitted.

| model | intact | PCG swapped | ECG swapped |
|---|---|---|---|
| `cross_attn_resnet18` | 0.938 +/- 0.039 | 0.913 +/- 0.049 (drop 0.024 +/- 0.018, 4/5) | 0.525 +/- 0.031 |
| `cross_attn_fusion` | 0.851 +/- 0.081 | 0.787 +/- 0.070 (drop 0.064 +/- 0.031, 5/5) | 0.545 +/- 0.050 |

The best model is nearly an ECG-only model: the wrong patient's PCG costs 0.024
AUC, the wrong ECG costs everything. Caveat for the paper: a swapped pair is
off-distribution for the cross-attention, so the drop bounds the model's
reliance on the matching PCG rather than measuring the information PCG carries;
a model trained without PCG could compensate. The clean test is a ResNet-18
ECG-only baseline, which was never trained.

Also checked ad hoc: an equal-weight average of the `ecg_only` and `pcg_only`
record probabilities (nothing fitted) gives patient AUC 0.840 +/- 0.093 against
0.864 for `ecg_only`, lower in 4 of 5 folds.

## ResNet-18 unimodal baselines (added 2026-09-11)

`resnet18_ecg_only` and `resnet18_pcg_only`: one branch of `cross_attn_resnet18`
(pretrained ResNet-18, 1x1 projection to 256, mean-pooled) and a
`Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 1)` head. Everything else in
the script is a byte-for-byte copy of the fusion model's, so the comparison
isolates exactly the second modality plus the cross-attention.

Why they were needed: the custom-CNN ablation compares fusion against unimodal
models with a *different* backbone from the best model, so it cannot say whether
`cross_attn_resnet18`'s +0.074 over `ecg_only` comes from the pretrained
backbone or from the fusion. The permutation test above says the fusion model
barely uses PCG, but a model trained without PCG might compensate - only these
baselines settle it.

Honesty note for the paper: they were added after the CV results were seen.
They are baselines trained with the fusion model's own recipe, not tuned models,
so post-hoc selection cannot favour them; state them as a follow-up ablation.

**Result** (record-level 5-fold CV, patient AUC, mean ± std):

| model | patient AUC | vs its counterpart, paired |
|---|---|---|
| `resnet18_ecg_only` | 0.930 ± 0.026 | +0.066 over `ecg_only` (wins 4-1, p=0.15) |
| `resnet18_pcg_only` | 0.741 ± 0.078 | +0.089 over `pcg_only` (wins 5-0, p=0.012) |
| `cross_attn_resnet18` | 0.938 ± 0.039 | **+0.008 over `resnet18_ecg_only`** (wins 2-3, p=0.50) |

Fusion adds nothing detectable at the ResNet level either; the best model's
+0.074 over `ecg_only` is ~0.066 backbone. Its one visible edge is specificity
(0.770 vs 0.733 at equal sensitivity 0.931), within fold noise. Consistent with
the permutation test (wrong-patient PCG costs 0.024).

## PCG pretraining on PhysioNet 2016 B-F (declared 2026-09-12, before results)

Why: fusion adds +0.008 over ResNet-18 ECG-only, and the PCG branch is the weak
side (ResNet-18 PCG-only 0.741). Subsets B-F are ~2,800 PCG-only recordings
from other hospitals and other patients - roughly 7x Training-A - so they can
strengthen the PCG branch without touching a CV test record.

Design, fixed before any B-F data was processed:

- `scripts/pretrain/01_convert.py` - B-F PCG with Training-A's normalisation
  and record QC; hard-fails on any Training-A record ID.
- `scripts/pretrain/02_scalogram.py` - fixed non-overlapping 3 s windows (no
  ECG, so not R-peak-centred), max 20 per record, Training-A's segment QC and
  transform (pinned byte-equal by a test).
- `scripts/pretrain/03_split.py` - record-level 85/15 train/val stratified by
  subset and label, for early stopping only.
- `scripts/modeling/pcg_pretrain/01_train.py` - `resnet18_pcg_only`'s model,
  30 epochs max, patience 5. Its checkpoint's `pcg_branch` initialises:
- `resnet18_pcg_only_pcgpre` and `cross_attn_resnet18_pcgpre` - generated from
  their twins; identical except the PCG initialisation. No ImageNet fallback.

Comparisons fixed in advance (paired per fold, patient AUC):
`resnet18_pcg_only_pcgpre` vs `resnet18_pcg_only`; `cross_attn_resnet18_pcgpre`
vs `resnet18_ecg_only` and vs `cross_attn_resnet18`.

Limitation to state: the pretrained branch saw un-centred windows from other
devices and sites; Training-A's are R-peak-centred.

## Decision threshold: validation-fitted screening points

`scripts/evaluation/06_screening_threshold.py`, on `cross_attn_resnet18`,
record-level mean aggregation. Per fold, the highest threshold reaching the
target sensitivity on the inner-validation records, applied to test:

| validation target | threshold (fold 0) | test sensitivity | test specificity |
|---|---|---|---|
| default 0.5 | 0.5 | 0.931 +/- 0.033 | 0.770 +/- 0.041 |
| 90% | 0.657 +/- 0.127 (0.598) | 0.903 +/- 0.032 | 0.829 +/- 0.076 |
| 95% | 0.428 +/- 0.147 (0.522) | 0.938 +/- 0.028 | 0.743 +/- 0.090 |
| 98% = 100% | 0.129 +/- 0.133 (0.378) | 0.983 +/- 0.022 | 0.486 +/- 0.200 |

**Decision:** 0.5 everywhere - the paper and the demo. The 100% target was
briefly deployed in the demo (fold 0: 0.378) and reverted the same day by the
owner; the sweep is reported as the operating-point trade-off. Moderate targets buy almost nothing; the 100% target buys ~5 points
of sensitivity for ~28 points of specificity, and its threshold is set by one
record per fold, so it swings from 0.0001 to 0.378. A threshold chosen on test
would have looked far better (0.31 gave 0.972 / 0.641) - that is the leak rule 5
exists to stop. Worth a sentence in the paper: operating-point selection needs
more validation data than five-fold CV on 405 records provides.

`tests/test_screening_threshold.py` pins the fitting rule.

## Key decisions

### The model definitions and the Dataset class are duplicated across training scripts, on purpose

This is a **decision, not an oversight.** The no-shared-utils convention means
each training script carries its own constants, its own `Dataset`, and its own
`nn.Module` definitions rather than importing from a sibling.

The payoff is the Kaggle offload: a Kaggle kernel is *that same script* with its
path constants pointed at `/kaggle/input/<slug>/` and `/kaggle/working/`. If the
models lived in a shared module, every kernel would need the module shipped and
`sys.path` patched, and the local and remote copies would drift apart. The
duplication buys a single source of truth for what actually runs.

The cost is real: a change to `CNNBranch` has to be made in ~7 files.
`tests/test_model.py` exists partly to catch a copy that drifted — it asserts
forward-pass output shapes for every architecture independently.

### Resume-safety is load-bearing, not aspirational

With dozens of queued runs across Kaggle kernels, the session will be
interrupted. Every training script checks for an existing checkpoint and an
already-completed MLflow run for the same (config, protocol, fold) and skips or
resumes rather than restarting.

### AMP is conditional

`torch.cuda.is_available()` gates autocast and `GradScaler`. Plain fp32
otherwise. This is what lets the identical file smoke-test on local CPU and then
run on a Kaggle T4.

## Data notes & gotchas
- (pending the modeling stage)

## Pending
- Everything.

## Ideas
- (none yet)
