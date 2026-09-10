# Task log 4 — Interpretability

## Scope
Contribution 2: Grad-CAM on `cross_attn_fusion`, and an honest read of whether
the attention lands where cardiac physiology says it should.

## Completed
- (pending the trained headline model)

## Key decisions
- **Grad-CAM is hand-rolled**, not taken from the `grad-cam` package. It is ~40
  lines of forward/backward hooks on the last `Conv2d` of each branch's
  `CNNBranch.features`, and keeping it inline preserves the
  self-contained-script convention that lets an evaluation script run unmodified
  inside a Kaggle kernel. The package would add a `ttach` dependency and a
  version-pinning surface for no functional gain here.
- Overlays go on the `jet` PNG renders, not on the uint8 memmap, so the figure is
  legible to a reader.

## Data notes & gotchas
- (pending)

## Pending
- The interpretation itself. The question to answer plainly: **does the ECG map
  concentrate on the QRS complex's time–frequency region, and the PCG map on the
  S1/S2 bursts?** If it does not, report that. A negative interpretability
  result is still a result, and inventing agreement is worse than not having it.
- Frame Contribution 2 as an *extension* of an active line (Alqudah et al. 2025
  already did Grad-CAM/SHAP/IG on dual-branch ECG–PCG with cross-modal
  attention), specifically validation on dual-CWT scalograms — not as a first.

## Ideas
- (none yet)
