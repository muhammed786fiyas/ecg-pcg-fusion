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
