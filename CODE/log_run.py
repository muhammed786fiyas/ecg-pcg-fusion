import os
import wfdb
import scipy.io as sio
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
LOG_FILE = "PROJECT_LOG.md"

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
# MAIN
# ============================================================
if __name__ == "__main__":
    log_raw_data()
    log_mat_data()
    log_train_test_split()
    log_segmentation()
    print("✅ Project log updated (raw → mat → split → segmentation)")
