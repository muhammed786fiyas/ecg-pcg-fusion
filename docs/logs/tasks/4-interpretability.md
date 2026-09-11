# Task log 4 — Interpretability

## Scope
Contribution 2: Grad-CAM on `cross_attn_fusion`, and an honest read of whether
the attention lands where cardiac physiology says it should.

## Completed
- Grad-CAM run on `cross_attn_fusion`, CV fold 0, 12 segments spanning correctly
  classified normal, correctly classified abnormal, false positives and false
  negatives. Figures in `reports/gradcam/`, per-segment numbers in
  `gradcam_summary.csv`.

## Result — a MIXED answer, reported as such

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

- **Grad-CAM on `cross_attn_resnet18`, the strongest model.** Day 1's analysis
  used `cross_attn_fusion`, which showed no fusion benefit. A single demo window
  from the ResNet model put ECG attention near 5 Hz (as before) but PCG attention
  near 16 Hz - below the S1/S2 band where `cross_attn_fusion` attended (median
  28 Hz). One window from a misclassified example is a lead, not a result; run
  the full fold-wise analysis in Hz before saying anything about it. Note the
  ResNet map is 7 x 7, coarser than the custom CNN's 14 x 14.

- (none yet)
