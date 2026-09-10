# Task log 2 — Features (CWT scalograms)

## Scope
Segment `.npz` → CWT magnitude scalogram → 224×224 uint8 memmap, per wavelet
config. Plus the `jet`-colormap PNG renders used only for paper figures and
Grad-CAM overlays.

## Completed
- Verified the wavelet facts directly against the installed stack (PyWavelets
  1.8.0, NeuroKit2 0.2.7, numpy 1.26.4) before writing the feature stage. See
  "`db4` -> `gaus4`" and "Measured frequency bands" below.

## Key decisions

### Store the model input as a uint8 memmap, not a PNG

One memmap per `(wavelet_config, modality)` of shape `(N, 224, 224)`, plus a
row-index CSV mapping `segment_variant_id -> row`.

The old pipeline rendered each scalogram through matplotlib's `jet` colormap
into a 3-channel PNG. That is wrong three times over:

1. It **triples the data** — a scalar magnitude field stored as RGB.
2. It **discards precision** through a perceptually non-uniform colour mapping.
3. It makes **`plt.savefig` plus PNG decode the dominant cost of every training
   epoch**, which on a CPU-bound loop is the single largest available speedup.

A uint8 memmap loads essentially for free and keeps the scalogram as the scalar
field it actually is. Expect roughly 450 MB per modality per config.

`jet` PNGs are still rendered — separately, and only for the ~30 segments needed
for paper figures and Grad-CAM overlays, into
`reports/figures/scalogram_examples/`.

Models take **1-channel** input. The pretrained ResNet-18 branch replicates the
single channel to 3 at load time (standard grayscale→pretrained-RGB practice).

### Reflect-pad before the CWT — a defect found by actually looking at a figure

**This was a real bug, caught late, and it is the most important note in this
file.**

The first paper figure of an ECG scalogram came out almost entirely dark. The
PCG panel showed clean S1/S2 bursts; the ECG panel showed two bright vertical
bands at the left and right edges and near-black everywhere in between.

Measured on the generated data:

| | ECG | PCG |
|---|---|---|
| edge / interior energy ratio (single segment) | **19.3** | ~1.0 |
| interior mean after min/max scaling | **4.6 / 255** | 32 / 255 |
| fraction of pixels below 26/255 | **85%** | 58% |

**Cause.** The CWT cone of influence. A wavelet at scale *s* has effective
support of a few times *s*. The ECG scales run to 500, so at the top of the
range the wavelet is as long as the 6000-sample window itself and the boundary
effect swamps the whole transform. PCG is unaffected because its scales stop at
130.

The damage was compounded by the per-image min/max normalization: the edge
artifact set the maximum, so the real cardiac content was compressed into the
bottom few percent of the 8-bit range.

**Fix.** Reflect-pad the signal by `PAD_SCALE_FACTOR * max(scales)` samples
before transforming, then crop the magnitude back to the real window. Measured
sweep on one segment:

| pad (samples) | edge/interior | interior mean | frac < 26/255 |
|---|---|---|---|
| 0 | 19.29 | 4.6 | 0.847 |
| 500 | 3.34 | 7.5 | 0.933 |
| **1000** | **0.63** | **59.3** | **0.348** |
| 2000 | 0.55 | 59.3 | 0.360 |
| 3000 | 0.55 | 59.3 | 0.360 |

Converged by ~1000; `PAD_SCALE_FACTOR = 4` gives 2000 for ECG and 520 for PCG,
which is a safe margin at ~1.5x the CWT cost. **Interior dynamic range improves
about 13-fold.**

**Why this justified discarding work.** It was found after 4 GPU kernels and 3
scalogram configs had completed, all of which were thrown away and regenerated.
That was still the cheap moment to find it, because the bug would not have
merely cost accuracy — **it would have produced a false scientific conclusion.**
Grad-CAM (Contribution 2) would have highlighted the boundary artifact, and the
honest-looking write-up would have read "the model does not concentrate on the
QRS time–frequency region" as a physiological finding, when it was an artifact
of preprocessing. A negative interpretability result is worth reporting; a
negative result caused by one's own bug is not.

**Lesson worth keeping:** the numeric QC gates all passed. The stage checked for
constant images and non-finite values and found nothing wrong. It took rendering
one PNG and looking at it to see the problem.

### `db4` → `gaus4` substitution in the wavelet ablation

The original plan proposed `db4` as one of the four ablated mother wavelets.
**It cannot be used.** `pywt.cwt(signal, scales, "db4", ...)` raises:

```
AttributeError: 'pywt._extensions._pywt.Wavelet' object has no attribute 'complex_cwt'
```

because `db4` is a *discrete orthogonal* wavelet and is not a member of
`pywt.wavelist(kind="continuous")`, which is exactly:

```
['cgau1'...'cgau8', 'cmor', 'fbsp', 'gaus1'...'gaus8', 'mexh', 'morl', 'shan']
```

**`gaus4` replaces it** — a real-valued continuous wavelet in the same spirit as
`mexh`. This is a documented substitution, **not** a silent swap: it must appear
in the paper's Method section. Do not try to force `db4` through `pywt.cwt`.

### Scale ranges are held fixed across wavelets

ECG scales `np.arange(20, 501)` (≈0.5–40 Hz), PCG scales `np.arange(7, 131)`
(≈20–250 Hz), for every wavelet in the ablation. Retuning scales per wavelet is
a different and much larger experiment, and mixing the two would confound
Contribution 1.

## Data notes & gotchas

### Measured frequency bands — the brief's Hz annotation was off for ECG

The brief specifies ECG scales `np.arange(20, 501)` and annotates them
"≈0.5–40 Hz". That annotation is wrong. `pywt.scale2frequency` gives
`f = f_c / (scale · dt)`, and with `cmor1.5-1.0` (`f_c = 1.0`) at `fs = 2000 Hz`
this is `f = 2000 / scale`, so:

| Modality | Wavelet | Scales | **Measured band** | Brief said |
|---|---|---|---|---|
| ECG | `cmor1.5-1.0` | `arange(20, 501)`, 481 scales | **4.0 – 100.0 Hz** | ≈0.5–40 Hz |
| PCG | `morl` | `arange(7, 131)`, 124 scales | **12.5 – 232.1 Hz** | ≈20–250 Hz |

Confirmed by running `pywt.cwt` on a 3 s NeuroKit2-simulated ECG at 2000 Hz.

**The scales are kept as specified, and the annotation is corrected.** The
scales are the explicit, final part of the spec — and the brief separately
requires that scale ranges stay fixed across every wavelet in the ablation, so
changing them would confound Contribution 1. 4–100 Hz is in any case a sound ECG
band: QRS energy sits at roughly 5–40 Hz and the upper end captures the sharp
QRS transients. The PCG band is close to the stated figure.

**Use the measured numbers in the paper, not the brief's.** Quoting 0.5–40 Hz
for a transform that does not reach below 4 Hz is the kind of detail a reviewer
checks.

### Verified environment facts
- `pywt.wavelist(kind="continuous")` on PyWavelets 1.8.0 is exactly
  `['cgau1'...'cgau8', 'cmor', 'fbsp', 'gaus1'...'gaus8', 'mexh', 'morl', 'shan']`.
- `nk.ecg_peaks(..., method="pantompkins1985")` works on 3.11 / numpy 1.26.4.

## Pending
- Generate the default config, then the six ablation configs.

## Ideas
- (none yet)
