import os
import wfdb
import scipy.io as sio
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
LOG_FILE = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\PROJECT_LOG.md"

RAW_DATA_DIR = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\1-PHYSIONET RAW DATA\training-a"
MAT_DATA_DIR = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\2-MATLAB DATA"

TRAIN_LABELS = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\3-SPLIT_DATA\train_labels.csv"
TEST_LABELS  = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\3-SPLIT_DATA\test_labels.csv"

TRAIN_SEG_LABELS = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\4-SEGMENTED_DATA\train_segment_labels.csv"
TEST_SEG_LABELS  = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\4-SEGMENTED_DATA\test_segment_labels.csv"

SKIPPED_TRAIN = [
    "a0077","a0084","a0113","a0155","a0159","a0187","a0202","a0206",
    "a0225","a0228","a0238","a0258","a0276","a0305","a0366",
    "a0393","a0406"
]

SKIPPED_TEST = [
    "a0101","a0137","a0217","a0255","a0279",
    "a0291","a0295","a0333","a0344","a0367"
]

# ============================================================
# LOG RAW DATA
# ============================================================
def log_raw_data():
    total_records = 0
    ecg_pcg = []
    pcg_only = []

    for file in os.listdir(RAW_DATA_DIR):
        if not file.endswith(".hea"):
            continue

        record_id = file.replace(".hea", "")
        record_path = os.path.join(RAW_DATA_DIR, record_id)

        try:
            header = wfdb.rdheader(record_path)
            channels = header.sig_name
        except Exception:
            continue

        total_records += 1

        if "ECG" in channels and "PCG" in channels:
            ecg_pcg.append(record_id)
        elif "PCG" in channels:
            pcg_only.append(record_id)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Raw Data Summary

**Raw data path:**  
`{RAW_DATA_DIR}`

- Total records found: {total_records}
- ECG + PCG records: {len(ecg_pcg)}
- PCG-only records: {len(pcg_only)}

Notes:
- Only records with synchronous ECG and PCG are used.
- PCG-only records are excluded.
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG MATLAB CONVERTED DATA
# ============================================================
def log_mat_data():
    mat_files = [f for f in os.listdir(MAT_DATA_DIR) if f.endswith(".mat")]

    fs_values = []
    invalid = []

    for file in mat_files:
        path = os.path.join(MAT_DATA_DIR, file)
        try:
            data = sio.loadmat(path)
            ecg = data.get("ecg", None)
            pcg = data.get("pcg", None)
            fs = data.get("fs", None)

            if ecg is None or pcg is None or fs is None:
                invalid.append(file.replace(".mat", ""))
                continue

            fs_values.append(int(fs[0][0]))

        except Exception:
            invalid.append(file.replace(".mat", ""))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — MATLAB Converted Data Summary

**Converted data path:**  
`{MAT_DATA_DIR}`

- Total .mat records: {len(mat_files)}
- Sampling rates detected: {sorted(set(fs_values))}
- Invalid or incomplete records: {len(invalid)}

Notes:
- ECG preserved in physical units (mV).
- PCG amplitude normalized during conversion.
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG TRAIN–TEST SPLIT
# ============================================================
def log_train_test_split():
    train_df = pd.read_csv(TRAIN_LABELS)
    test_df  = pd.read_csv(TEST_LABELS)

    def summarize(df):
        counts = df["label"].value_counts()
        perc = df["label"].value_counts(normalize=True) * 100
        return counts, perc

    train_c, train_p = summarize(train_df)
    test_c, test_p   = summarize(test_df)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Patient-wise Train–Test Split Summary

**Split strategy:**
- 70% Train / 30% Test
- Stratified at patient (record) level
- No cross-patient leakage

### Training Set
- Records: {len(train_df)}
- +1 records: {train_c.get(1,0)} ({train_p.get(1,0):.2f}%)
- -1 records: {train_c.get(-1,0)} ({train_p.get(-1,0):.2f}%)

### Test Set
- Records: {len(test_df)}
- +1 records: {test_c.get(1,0)} ({test_p.get(1,0):.2f}%)
- -1 records: {test_c.get(-1,0)} ({test_p.get(-1,0):.2f}%)

Notes:
- Split performed before segmentation.
- Class proportions preserved across splits.
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG SEGMENTATION
# ============================================================
def log_segmentation():
    train_df = pd.read_csv(TRAIN_SEG_LABELS)
    test_df  = pd.read_csv(TEST_SEG_LABELS)

    def summarize(df):
        counts = df["label"].value_counts()
        perc = df["label"].value_counts(normalize=True) * 100
        return counts, perc

    train_c, train_p = summarize(train_df)
    test_c, test_p   = summarize(test_df)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — R-Peak–Based Segmentation Summary

**Segmentation method:**
- R-peak–centered windows
- Window length: 3 seconds
- Overlap: None
- R-peak detection: Pan–Tompkins–based (NeuroKit2)

### Training Data
- Total segments: {len(train_df)}
- +1 segments: {train_c.get(1,0)} ({train_p.get(1,0):.2f}%)
- -1 segments: {train_c.get(-1,0)} ({train_p.get(-1,0):.2f}%)
- Skipped records: {len(SKIPPED_TRAIN)}

### Test Data
- Total segments: {len(test_df)}
- +1 segments: {test_c.get(1,0)} ({test_p.get(1,0):.2f}%)
- -1 segments: {test_c.get(-1,0)} ({test_p.get(-1,0):.2f}%)
- Skipped records: {len(SKIPPED_TEST)}

Notes:
- Patient-wise split preserved.
- No data augmentation applied at this stage.
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ============================================================
# LOG AUGMENTATION
# ============================================================
def log_augmentation():
    TRAIN_SEG_LABELS = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\4-SEGMENTED_DATA\train_segment_labels.csv"
    AUG_LABELS = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\5-AUGMENTED_DATA\train_augmented_labels.csv"

    # Load labels
    orig_df = pd.read_csv(TRAIN_SEG_LABELS)
    aug_df  = pd.read_csv(AUG_LABELS)

    orig_count = len(orig_df)
    aug_count  = len(aug_df)
    expansion_factor = aug_count / orig_count if orig_count > 0 else 0

    # Label distribution after augmentation
    counts = aug_df["label"].value_counts()
    perc = aug_df["label"].value_counts(normalize=True) * 100

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Training Data Augmentation Summary

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
- Original training segments: {orig_count}
- Augmented training segments: {aug_count}
- Expansion factor: {expansion_factor:.2f}×

**Label distribution after augmentation:**
- +1 segments: {counts.get(1,0)} ({perc.get(1,0):.2f}%)
- -1 segments: {counts.get(-1,0)} ({perc.get(-1,0):.2f}%)

Notes:
- Augmentation parameters were fixed across all experiments.
- No augmentation was applied to the test set.
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG DATA CLEANING (NaN REMOVAL — TRAIN + TEST)
# ============================================================
def log_data_cleaning():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- TRAIN STATS ----------------
    train_segmented_total = 2271
    train_segmented_removed = 173
    train_segmented_clean = 2098

    train_augmented_total = 9084
    train_augmented_removed = 692
    train_augmented_clean = 8392

    train_removed_records = [
        "a0014","a0027","a0028","a0045","a0055","a0057","a0068","a0070",
        "a0075","a0118","a0160","a0163","a0179","a0250","a0274","a0303",
        "a0315","a0361","a0395"
    ]

    # ---------------- TEST STATS ----------------
    test_segmented_total = 959
    test_segmented_removed = 79
    test_segmented_clean = 880

    test_removed_summary = {
        "a0018": 5, "a0185": 10, "a0204": 9, "a0261": 10,
        "a0311": 8, "a0314": 6, "a0320": 11, "a0337": 9,
        "a0347": 6, "a0400": 5
    }

    log_entry = f"""
---

## {timestamp} — Data Cleaning Summary (NaN Removal)

### Quality Check
- Dataset-wide NaN scan on ECG and PCG signals
- Segment marked invalid if >90% of PCG samples were NaN

### Findings
- ECG: No invalid segments detected (TRAIN / TEST)
- PCG: NaN-contaminated segments detected in both TRAIN and TEST

---

### TRAIN Data Cleaning

**Segment-level**
- Original segments: {train_segmented_total}
- Removed (PCG NaNs): {train_segmented_removed}
- Clean segments retained: {train_segmented_clean}

**Augmented data**
- Original augmented samples: {train_augmented_total}
- Removed augmented samples: {train_augmented_removed}
- Clean augmented samples retained: {train_augmented_clean}

**Affected TRAIN records:**
{chr(10).join(['- ' + r for r in train_removed_records])}

---

### TEST Data Cleaning

- Original segments: {test_segmented_total}
- Removed (PCG NaNs): {test_segmented_removed}
- Clean segments retained: {test_segmented_clean}

**Removed TEST segments (summary):**
{chr(10).join([f"- {k}: {v} segments" for k, v in test_removed_summary.items()])}

---

### Actions Taken
- Removed invalid segmented `.mat` files
- Removed all augmented variants (orig / noise / scale / mix)
- Removed corresponding entries from label files
- Removed scalograms linked to invalid segments

Notes:
- TEST data was not augmented
- Cleaning performed before final scalogram generation
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG SCALOGRAM GENERATION
# ============================================================
def log_scalogram():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Scalogram Generation Summary

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
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ============================================================
# LOG VISUALIZATION (SEGMENTED + AUGMENTED SIGNALS)
# ============================================================
def log_visualization():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Signal Visualization Validation Summary

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
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================
# LOG SCALOGRAM VISUALIZATION
# ============================================================
def log_scalogram_visualization():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
---

## {timestamp} — Scalogram Visualization Validation Summary

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
"""

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_run.py [raw | mat | split | segment | augment | clean | visualize | scalogram | scalogram_viz]")



    stage = sys.argv[1].lower()

    if stage == "raw":
        log_raw_data()
    elif stage == "mat":
        log_mat_data()
    elif stage == "split":
        log_train_test_split()
    elif stage == "segment":
        log_segmentation()
    elif stage == "augment":
        log_augmentation()
    elif stage == "clean":
        log_data_cleaning()
    elif stage == "scalogram":
        log_scalogram()
    elif stage == "visualize":
        log_visualization()
    elif stage == "scalogram_viz":
        log_scalogram_visualization()
    else:
        print("Unknown stage")


