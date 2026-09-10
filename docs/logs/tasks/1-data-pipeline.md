# Task log 1 — Data pipeline

## Scope
`data/raw/physionet2016_training_a/` → per-record `.npz` → record QC →
R-peak-centred segments → segment QC → record-level fold assignment → seeded
4x augmentation.

## Completed
- Raw data verified by direct inspection before any code was written.

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

## Pending
- Run the pipeline and record the real counts at every gate.

## Ideas
- (none yet)
