# Task log 4 — Interpretability

## Scope
Contribution 2: Grad-CAM on the cross-attention models (`cross_attn_fusion`, and
from 2026-09-11 `cross_attn_resnet18`), and an honest read of whether the
attention lands where cardiac physiology says it should.

## Completed
- Grad-CAM run on `cross_attn_fusion`, CV fold 0, 12 segments spanning correctly
  classified normal, correctly classified abnormal, false positives and false
  negatives. Figures in `reports/gradcam/`, per-segment numbers in
  `gradcam_summary.csv`.

## Result, revised 2026-09-11 — against the right baselines, the frequency story does not hold

Re-run with `scripts/evaluation/02_gradcam.py` on both cross-attention models,
all five CV folds, EVERY test segment (3,752 per model, maps for the abnormal
logit), then mean ± std across folds. Two baselines added:

- **uniform-map share.** The image rows are evenly spaced in wavelet *scale*,
  and frequency goes as 1/scale, so the axis is hyperbolic in Hz: 62.5% of the
  ECG rows lie between 4 and 10 Hz (scales 200-500 of 20-500), and 52.7% of the
  PCG rows between 12.5 and 25 Hz. A map that prefers nothing already scores
  that. `tests/test_gradcam.py` pins the 62.5%.
- **scalogram-energy share.** Where the image's own intensity sits - what a map
  that simply follows bright pixels would score.

| model | modality | band | Grad-CAM share | uniform | energy | CAM / uniform | CAM / energy |
|---|---|---|---|---|---|---|---|
| `cross_attn_resnet18` | ECG | 4-10 Hz | 0.683 ± 0.085 | 0.625 | 0.790 | 1.09 | 0.86 |
| | ECG | 10-25 Hz | 0.230 ± 0.075 | 0.250 | 0.184 | 0.92 | 1.25 |
| | PCG | 12.5-25 Hz | 0.613 ± 0.063 | 0.527 | 0.607 | 1.16 | 1.01 |
| | PCG | 25-50 Hz | 0.220 ± 0.046 | 0.263 | 0.313 | 0.83 | 0.70 |
| `cross_attn_fusion` | ECG | 4-10 Hz | 0.813 ± 0.150 | 0.625 | 0.783 | 1.30 | 1.04 |
| | ECG | 10-25 Hz | 0.104 ± 0.090 | 0.250 | 0.191 | 0.42 | 0.55 |
| | PCG | 12.5-25 Hz | 0.703 ± 0.124 | 0.527 | 0.607 | 1.34 | 1.16 |
| | PCG | 25-50 Hz | 0.142 ± 0.043 | 0.263 | 0.313 | 0.54 | 0.45 |

Full tables, with the 50+ Hz band and the normal/abnormal split:
`reports/gradcam/<family>/gradcam_cross_fold.md`.

What this says:

1. **Neither model shows a clear physiological frequency preference.** The
   ResNet-18 maps are close to flat against geometry (ratios 0.7-1.2). The
   custom CNN leans towards the lowest ECG band (1.30x uniform), but that is
   exactly where the scalogram's own energy sits (1.04x energy): its attention
   follows brightness.
2. **Day 1's "PCG yes, ECG no" is withdrawn.** Its "93.6% of ECG mass below
   10 Hz" was read against zero - against 62.5% for a map that prefers nothing,
   and ~79% for one that follows brightness. Its PCG median of 28 Hz "in the
   S1/S2 band" sits where a uniform map's median already falls (~24 Hz). Both
   halves were mostly axis geometry. The QRS / S1-S2 story can be neither
   claimed nor refuted from these maps.
3. **What does hold is time-localisation.** The custom CNN's maps are 5.7x
   (ECG) and 5.1x (PCG) more concentrated in time than a flat map - they lock
   onto discrete events in the window. ResNet-18's are ~2x: its 7 x 7 map spans
   ~0.43 s per cell, too coarse to localise a heart sound.
4. **ResNet-18's ECG maps are empty for 36% of normal segments** (389/1087),
   against 2% of abnormal ones (58/2665): for many normal segments nothing in
   the ECG pushes towards "abnormal" - what a working abnormal-logit map should
   show. Empty maps are excluded from the averages, not averaged in as zeros.
5. The day-2 demo lead, "ResNet PCG attention near 16 Hz", was one window;
   across all segments its PCG median is 25.7 ± 2.7 Hz, the same as the custom
   CNN's 25.9 ± 10.5 Hz.

**For the paper:** report Grad-CAM-on-scalogram frequency attributions only
against these baselines. The methodological point earns its own paragraph: a
CWT scalogram's row axis is non-uniform in Hz, so raw Grad-CAM band shares
mostly restate the transform's geometry. Median-frequency summaries have the
same problem (uniform-map median ~7.7 Hz ECG, ~24 Hz PCG).

## Day-1 result (SUPERSEDED 2026-09-11 - read against zero, not against the axis geometry)

The question §10.2 asks: does the ECG map concentrate on the QRS complex's
time–frequency region, and the PCG map on the S1/S2 bursts?

Grad-CAM mass by frequency band, mean over the 12 selected segments:

| modality | 0–10 Hz | 10–25 Hz | 25–50 Hz | 50+ Hz | peak | median | time concentration |
|---|---|---|---|---|---|---|---|
| ECG | **0.936** | 0.047 | 0.017 | 0.000 | 7.3 Hz | 6.0 Hz | 5.6× |
| PCG | 0.000 | **0.499** | 0.282 | 0.218 | 62.0 Hz | 28.1 Hz | 7.3× |

**PCG — yes.** Attention sits squarely in the S1/S2 band (median 28 Hz, peak
62 Hz; heart sounds are classically ~20–150 Hz) and is strongly time-localised
(7.3× concentration), i.e. locked onto discrete bursts rather than smeared
across the window. This is the expected physiological behaviour.

**ECG — not as claimed.** 93.6% of the mass sits below 10 Hz, peaking at 7.3 Hz.
QRS energy is classically 5–40 Hz, and the sharp QRS transient lives nearer
10–40 Hz. The model is attending to the **low-frequency envelope** of the
cardiac cycle — the QRS-T complex as a whole, and plausibly the T wave — rather
than to the QRS spike specifically. It *is* time-locked (5.6× concentration), so
it is tracking cardiac events; it is the frequency band that does not match the
QRS story.

**So the honest summary is: the PCG branch behaves as the physiological account
predicts, and the ECG branch does not.** Reporting "the model attends to QRS and
S1/S2" would be half right and half wishful.

### Caveats that belong with this result

1. **The ECG transform starts at 4 Hz**, so the "0–10 Hz" bin is really 4–10 Hz
   and sits at the very edge of the represented range. A transform reaching
   lower would be needed to say where the mass truly peaks.
2. **The CAM may partly be following image energy.** The ECG scalogram's own
   energy is concentrated at low frequency (visible in
   `reports/figures/scalogram_examples/`), so attention there is not
   independent evidence of a learned preference.
3. **One fold, 12 segments, one architecture.** `cross_attn_fusion` is also the
   model that showed no advantage over ECG-only, so this is interpretability of
   a model that is not the strongest one. `cross_attn_resnet18` would need
   separate hooks (its branches are ResNet, not `CNNBranch`).

### A labelling bug that would have inverted this conclusion

The first version binned CAM rows into "low/mid/high thirds" **by row index** and
named them that way. Row 0 is the *smallest scale*, which is the *highest*
frequency — so the field called `freq_low_third` actually held high-frequency
mass. Reading it naively gave "ECG mass is concentrated at high frequency",
i.e. exactly the QRS story we wanted, and exactly backwards.

`cam_mass_profile()` now maps each row back through its scale to Hz via
`pywt.scale2frequency` and reports named bands in Hz. **The conclusion reversed
once the axis was labelled correctly.**

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

- *Done 2026-09-11:* Grad-CAM on `cross_attn_resnet18` across all folds - see
  the revised result above. The single-window "PCG near 16 Hz" lead did not
  hold (25.7 Hz across all segments).
- A frequency-resolved attribution that is not tied to the scale-axis geometry:
  e.g. occlusion of fixed-Hz bands (mask 10-25 Hz, measure the logit change).
  That asks the physiological question directly instead of through row counts.

- (none yet)
