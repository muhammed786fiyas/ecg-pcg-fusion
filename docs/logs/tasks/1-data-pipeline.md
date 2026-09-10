# Task log 1 — Data pipeline

## Scope
`data/raw/physionet2016_training_a/` → per-record `.npz` → record QC →
R-peak-centred segments → segment QC → record-level fold assignment → seeded
4x augmentation.

## Completed
- Raw data verified by direct inspection before any code was written.
- Stages `01`–`06` written and run end-to-end on the full dataset. Real counts
  at every gate below.

## Actual counts at every gate

| Gate | In | Out | Dropped | Notes |
|---|---|---|---|---|
| `01_convert` | 409 records (REFERENCE.csv) | **405** | 4 | exactly the declared PCG-only set — the script hard-fails if the excluded set differs |
| `02_record_qc` | 405 | **405** | 0 | no record is majority-NaN, flat, or under 3 s |
| `03_segment` | 405 records | **3752 segments** | 0 records | every record yielded ≥1 window |
| `04_segment_qc` | 3752 segments | **3752** | 0 | 194 segments repaired by interpolation |
| `05_assign_folds` | 405 records | 405 assigned | — | dev 243/60/102; CV 5 × (275 train / 49 inner-val / 81 test) |
| `06_augment` | 3752 segments | **15008 rows** | — | 4 variants × 3752, all finite |

**Label balance.** REFERENCE.csv is 117 normal / 292 abnormal. All 4 PCG-only
records are abnormal, so the 405 kept records are **117 normal / 288 abnormal**
(71.1% abnormal). At segment level: 1087 normal / 2665 abnormal. The imbalance is
substantial and is handled with `pos_weight` fitted **inside each fold's training
portion**, never globally.

**Segments per record**: min 1, median 10, max 11, mean 9.26.
**R-peaks per record**: min 4, median 52, max 105.
Records are ~35 s at 2000 Hz, so ~11 non-overlapping 3 s windows is the ceiling.

**Fold balance** — abnormal share in each CV test fold: 0.716, 0.716, 0.716,
0.704, 0.704. Stratification is holding.

### The segment-QC result differs from what the brief expected — and why

The brief expected segment QC to drop "~173 train and ~79 test segments", the
order of magnitude the old pipeline saw. **We drop zero.** That is not a bug and
the threshold was not softened:

- The specified gate is ">90% NaN in either channel". No segment comes close.
- 194 segments across 44 records *do* contain some non-finite samples, but the
  worst case is **0.47% of ECG samples** and **1.68% of PCG samples**. These are
  repaired by linear interpolation, which the brief also specifies, and then
  re-checked for any surviving NaN/Inf.
- The old pipeline's larger drop count most likely came from a stricter rule
  (dropping on *any* NaN) rather than from different data.

Counts are in `data/interim/segment_qc_repaired.csv` and
`segment_qc_dropped.csv`. Reported as measured.

## Data notes & gotchas

**Verified raw facts (counted from the folder, not assumed):**
- 409 `.hea` headers, 405 `.dat` files.
- The 4 records with a header but no `.dat` are **a0041, a0117, a0220, a0233**.
  Their headers declare a single PCG channel and no ECG channel, e.g.
  `a0041 1 2000 70218` / `a0041.wav 16+44 1 16 0 0 0 0 PCG`. These are the
  PCG-only records and are excluded — the project is entirely about synchronous
  dual-modality input. **405 usable records**, matching the brief.
- Dual-modality headers declare two channels, e.g. `a0001 2 2000 71332` with
  `a0001.wav ... PCG` and `a0001.dat ... ECG`. Sampling rate is **2000 Hz** for
  every record.
- The `.wav` files duplicate the PCG channel and are ignored; `wfdb` reads both
  channels straight from the header.
- Labels come from `REFERENCE.csv` (409 rows, `record_id,label`, label ∈ {-1, +1}),
  mapped to {0, 1} = {normal, abnormal}.

## Key decisions

- **`record_id` is treated as the patient identifier.** PhysioNet 2016
  Training-A ships as a flat set of recordings with no subject-ID field, and the
  literature treats it as one record per subject. This is recorded as an
  **assumption, not a verified fact**: if two records came from the same person,
  record-level splitting is not patient-level splitting and the entire protocol
  rests on it. The dataset ships nothing that would let us check. Stated the same
  way in `PROJECT_CONTEXT.md`, and to be stated the same way in the paper.

- **Record-level QC runs before splitting.** The old pipeline filtered only per
  segment, so records that silently contributed zero segments still counted
  toward fold sizes and skewed the class balance. Gating at record level first
  means folds are assigned over records that actually survive.

- **Segmentation runs before fold assignment.** Safe because segmentation is
  per-record deterministic with no RNG — a record's segments are byte-identical
  whichever fold it later lands in — and it means fold sizes reflect the records
  that actually survive QC rather than being degraded by post-hoc dropouts.

- **PCG is normalized per record by its own max absolute value**, so no
  cross-record statistic is ever computed and there is nothing to leak.

- **`04_segment_qc` writes to a new directory instead of repairing `03`'s output
  in place.** The first version interpolated NaNs in place. That was wrong twice:
  DVC cannot track a stage whose output *is* its dependency, and — worse — after
  one run the evidence of what had been repaired was gone, so the stage could not
  be re-inspected. Caught when a follow-up NaN audit came back implausibly clean.
  `03`'s output is now immutable and `04` writes `data/interim/segments_qc/`.

- **Augmentation seeds per (segment, variant), not per segment.** The brief's
  example hashes `f"{seed}:{segment_id}"`. Hashing the variant in as well gives
  each variant its own independent stream, so adding or reordering variants
  later cannot silently change the values of the existing ones. Same
  reproducibility argument, one level down.

- **The `_combined` PCG time shift pads with the edge value rather than using
  `np.roll`.** Rolling wraps the tail of the window onto the front, splicing
  together two moments that were never adjacent — an artefact, not a plausible
  misalignment.

- **Reproducibility verified, not assumed.** Re-deriving any variant from its
  seed reproduces the stored file byte-for-byte, `_orig` is byte-identical to its
  source segment, and distinct (segment, variant) pairs get distinct seeds.
  Locked in as `tests/test_reproducibility.py`.

## Pending
- Run the pipeline and record the real counts at every gate.

## Ideas
- (none yet)
