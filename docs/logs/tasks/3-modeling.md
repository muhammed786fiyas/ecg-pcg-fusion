# Task log 3 — Modeling

## Scope
Seven architectures (`ecg_only`, `pcg_only`, `dual_cnn`, `warm_start_fusion`,
`cbam_fusion`, `cross_attn_fusion`, `cross_attn_resnet18`), the training loop,
and the split-protocol negative control.

## Completed
- (pending the modeling stage)

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
