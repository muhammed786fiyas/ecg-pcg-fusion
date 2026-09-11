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

**Decision:** the demo deploys the 100% target (fold 0: 0.378); the paper's
headline numbers stay at 0.5 and the sweep is reported as the operating-point
trade-off. Moderate targets buy almost nothing; the 100% target buys ~5 points
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
