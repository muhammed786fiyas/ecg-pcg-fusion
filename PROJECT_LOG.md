# Project Log

## Project Title
**Synchronous ECG–PCG Based Cardiac Abnormality Detection**

---

## Dataset

- **Source:** PhysioNet Challenge 2016 (Kaggle release)
- **Subset used:** Training-A
  - Only subset containing synchronous ECG and PCG signals
  - Training-B to Training-F were excluded (PCG-only)
- **Task:** Binary classification `{+1, -1}`

---

## Data Preparation

### Raw Data Format
- `.hea` — header file (metadata, channel names)
- `.dat` — signal data (WFDB format)
- `.wav` — PCG audio (not used to avoid duplication)

### Converted Data
- Converted WFDB records to `.mat` format
- Each `.mat` file contains:
  - `ecg` — ECG signal (physical units, mV)
  - `pcg` — PCG signal (amplitude-normalized to `[-1, 1]`)
  - `fs` — sampling frequency

---

## Step 1: Patient-wise Train–Test Split

- **Split ratio:** 70% Train / 30% Test
- **Method:** Stratified split at record (patient) level
- **Training records:** 283
- **Testing records:** 122
- **Outputs:**
  - `train/`, `test/` record-level `.mat` folders
  - `train_labels.csv`, `test_labels.csv`
- **Notes:**
  - Label proportions preserved
  - No data leakage

---

## Step 2: ECG–PCG Segmentation

### Segmentation Strategy
- **Method:** R-peak–centered segmentation
- **Window length:** 3 seconds
- **Overlap:** Non-overlapping windows
- **R-peak detection algorithm:**
  - Pan–Tompkins–based method (NeuroKit2)
  - Chosen for robustness and physiological relevance
- **Handling failures:**
  - Records with unreliable R-peak detection are skipped

---

### Training Segmentation Results
- **Total segments:** 2,271
- **Label distribution:**
  - `+1`: 1,598 (70.37%)
  - `-1`: 673 (29.63%)
- **Skipped training records:**
a0077, a0084, a0113, a0155, a0159, a0187, a0202, a0206,a0225, a0228, a0238, a0258, a0276, a0305, a0366, a0393, a0406

- **Outputs:**
  - Segmented `.mat` files
  - `train_segment_labels.csv`

---

### Test Segmentation Results
- **Total segments:** 959
- **Label distribution:**
  - `+1`: 649 (67.67%)
  - `-1`: 310 (32.33%)
- **Skipped test records:**
a0101, a0137, a0217, a0255, a0279, a0291, a0295, a0333, a0344, a0367

- **Outputs:**
  - Segmented `.mat` files
  - `test_segment_labels.csv`

---

## Dataset Expansion Summary

- **Training:** 283 records → 2,271 segments (~8× increase)
- **Testing:** 122 records → 959 segments (~7.9× increase)
- **Label handling:** Segment inherits record-level label
- **Data leakage:** Prevented via patient-wise split

---

## Environment

- **Python:** 3.9
- **Libraries:**
  - `wfdb`
  - `scipy`
  - `numpy`
  - `pandas`
  - `neurokit2==0.2.7`

---

## Current Status

- [x] Raw data conversion
- [x] Patient-wise train–test split
- [x] R-peak–based segmentation
- [x] Segment-level label generation
- [x] Dataset statistics documented

---

## Next Steps

- [ ] Step 3: Data augmentation (TRAIN only)
- [ ] Step 4: Scalogram generation
- [ ] Step 5: Model training
- [ ] Step 6: Evaluation on test set
- [ ] Step 7: Results analysis and reporting

---

## Notes / Decisions Log

- Training-A only used due to synchronous ECG–PCG availability
- ECG kept in physical units; PCG normalized
- Records with poor ECG quality skipped during segmentation
- Python version fixed at 3.9 for stability

---

## ✅ How to use this template

- Update Current Status as you progress
- Append decisions in Notes / Decisions Log
- Reuse the structure for future projects

---

## 🔄 Automated Project Log (Append Only)

> ⚠️ Entries below this line are generated automatically.  
> Do not edit manually.



---

## 2026-01-17 15:28:17 — Raw Data Summary

**Raw data path:**  
`E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\1-PHYSIONET RAW DATA\training-a`

- Total records found: 409
- ECG + PCG records: 405
- PCG-only records: 4

Notes:
- Only records with synchronous ECG and PCG are used.
- PCG-only records are excluded.

---

## 2026-01-17 15:28:24 — MATLAB Converted Data Summary

**Converted data path:**  
`E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\2-MATLAB DATA`

- Total .mat records: 405
- Sampling rates detected: [2000]
- Invalid or incomplete records: 0

Notes:
- ECG preserved in physical units (mV).
- PCG amplitude normalized during conversion.

---

## 2026-01-17 15:28:24 — Patient-wise Train–Test Split Summary

**Split strategy:**
- 70% Train / 30% Test
- Stratified at patient (record) level
- No cross-patient leakage

### Training Set
- Records: 283
- +1 records: 201 (71.02%)
- -1 records: 82 (28.98%)

### Test Set
- Records: 122
- +1 records: 87 (71.31%)
- -1 records: 35 (28.69%)

Notes:
- Split performed before segmentation.
- Class proportions preserved across splits.

---

## 2026-01-17 15:28:24 — R-Peak–Based Segmentation Summary

**Segmentation method:**
- R-peak–centered windows
- Window length: 3 seconds
- Overlap: None
- R-peak detection: Pan–Tompkins–based (NeuroKit2)

### Training Data
- Total segments: 2271
- +1 segments: 1598 (70.37%)
- -1 segments: 673 (29.63%)
- Skipped records: 17

### Test Data
- Total segments: 959
- +1 segments: 649 (67.67%)
- -1 segments: 310 (32.33%)
- Skipped records: 10

Notes:
- Patient-wise split preserved.
- No data augmentation applied at this stage.


---

## 2026-01-17 16:05:22 — Training Data Augmentation Summary

**Augmentation stage:**
- Applied after segmentation and before scalogram generation
- Applied to training data only
- Signal-domain augmentation

**Augmentation methods:**

- **ECG**
  - Additive Gaussian noise (σ = 1% of ECG standard deviation)
  - Amplitude scaling (factor ∈ [0.9, 1.1])

- **PCG**
  - Additive Gaussian noise (σ = 0.02)
  - Amplitude scaling (factor ∈ [0.85, 1.15])
  - Temporal shift (±50 ms)

**Dataset size:**
- Original training segments: 2271
- Augmented training segments: 9084
- Expansion factor: 4.00×

**Label distribution after augmentation:**
- +1 segments: 6392 (70.37%)
- -1 segments: 2692 (29.63%)

Notes:
- Augmentation parameters were fixed across all experiments.
- No augmentation was applied to the test set.


---

## 2026-01-18 05:52:28 — Data Cleaning Summary (NaN Removal)

### Quality Check
- Dataset-wide NaN scan on ECG and PCG signals
- Segment marked invalid if >90% of PCG samples were NaN

### Findings
- ECG: No invalid segments detected (TRAIN / TEST)
- PCG: NaN-contaminated segments detected in both TRAIN and TEST

---

### TRAIN Data Cleaning

**Segment-level**
- Original segments: 2271
- Removed (PCG NaNs): 173
- Clean segments retained: 2098

**Augmented data**
- Original augmented samples: 9084
- Removed augmented samples: 692
- Clean augmented samples retained: 8392

**Affected TRAIN records:**
- a0014
- a0027
- a0028
- a0045
- a0055
- a0057
- a0068
- a0070
- a0075
- a0118
- a0160
- a0163
- a0179
- a0250
- a0274
- a0303
- a0315
- a0361
- a0395

---

### TEST Data Cleaning

- Original segments: 959
- Removed (PCG NaNs): 79
- Clean segments retained: 880

**Removed TEST segments (summary):**
- a0018: 5 segments
- a0185: 10 segments
- a0204: 9 segments
- a0261: 10 segments
- a0311: 8 segments
- a0314: 6 segments
- a0320: 11 segments
- a0337: 9 segments
- a0347: 6 segments
- a0400: 5 segments

---

### Actions Taken
- Removed invalid segmented `.mat` files
- Removed all augmented variants (orig / noise / scale / mix)
- Removed corresponding entries from label files
- Removed scalograms linked to invalid segments

Notes:
- TEST data was not augmented
- Cleaning performed before final scalogram generation

---

## 2026-01-18 05:53:49 — Scalogram Generation Summary

**Input data:**
- Cleaned augmented ECG–PCG segments
- Invalid segments excluded prior to scalogram generation

**Time–frequency representation:**
- Continuous Wavelet Transform (CWT)

**Wavelet configuration:**
- ECG:
  - Wavelet: Complex Morlet (cmor1.5-1.0)
  - Scales: 20–500 (≈0.5–40 Hz)
- PCG:
  - Wavelet: Real Morlet (morl)
  - Scales: 7–130 (≈20–250 Hz)

**Image settings:**
- Output size: 224 × 224 pixels
- Format: PNG
- Separate directories for ECG and PCG

**Processing details:**
- Scalogram generation is resume-safe (skip-if-exists enabled)
- Partial generation supported
- No downsampling applied
- Consistent configuration used across the dataset

Notes:
- Scalograms corresponding to removed segments were deleted
- Final dataset integrity preserved before model training

---

## 2026-01-18 17:55:33 — Signal Visualization Validation Summary

### Visualization Scope
- Time-domain visualization of segmented ECG–PCG signals
- Time-domain comparison of original vs augmented ECG–PCG signals
- Visualization performed on representative samples from training data

---

### Segmented Signal Validation

**Checks performed:**
- Visual inspection of ECG morphology and QRS complexes
- Verification of R-peak–centered segmentation
- Inspection of PCG heart sound bursts (S1/S2)
- Verification of ECG–PCG temporal synchrony

**Observations:**
- ECG segments show clear QRS complexes with preserved morphology
- R-peaks are consistently located near the center of each segment
- PCG signals exhibit distinct, physiologically plausible heart sound patterns
- No flatlines, clipping, or NaN-dominated segments observed

---

### Augmented Signal Validation

**Augmentation types inspected:**
- Additive Gaussian noise
- Amplitude scaling
- Signal mixing

**Observations:**
- Noise augmentation introduces low-amplitude perturbations without distorting ECG or PCG morphology
- Amplitude scaling preserves waveform shape with uniform gain variation
- Signal mixing increases variability while maintaining physiological plausibility
- Temporal alignment between ECG and PCG signals is preserved across all augmentations

---

### Decision
- Segmented and augmented ECG–PCG signals verified to be physiologically valid
- All augmentation strategies retained
- Data approved for scalogram generation and model training

Notes:
- Visualization performed as a manual quality-control step
- Validation results logged to ensure reproducibility and traceability

---

## 2026-01-18 18:10:51 — Scalogram Visualization Validation Summary

### Visualization Scope
- Manual inspection of ECG and PCG scalograms generated using CWT
- Comparison of original and augmented scalograms
- Validation performed prior to CNN-based model training

---

### Axis Interpretation Verification
- X-axis corresponds to time (0–3 seconds per segment)
- Y-axis corresponds to wavelet scale (inverse of frequency)
- Color intensity represents localized signal energy

---

### ECG Scalogram Observations
- Dominant energy localized in lower scales, consistent with ECG physiological bandwidth
- Clear vertical energy ridges corresponding to QRS complexes
- Expected boundary artifacts observed at segment edges
- No empty, saturated, or corrupted scalograms detected

---

### PCG Scalogram Observations
- Time-localized, high-energy vertical patterns corresponding to heart sounds (S1/S2)
- Broader frequency distribution compared to ECG, reflecting acoustic characteristics
- Stable background with no artificial banding or noise-dominated regions

---

### Augmented Scalogram Assessment
- Noise augmentation introduces mild texture variation without altering dominant time–frequency structure
- Amplitude scaling preserves scalogram structure with proportional intensity change
- Signal mixing increases pattern diversity while maintaining physiological plausibility
- Temporal alignment between ECG and PCG scalograms preserved

---

### Decision
- ECG and PCG scalograms verified to be physiologically meaningful
- Augmented scalograms retained without modification
- Dataset approved for CNN training

Notes:
- Scalogram validation performed as a quality-control step
- Results logged to ensure reproducibility and auditability
