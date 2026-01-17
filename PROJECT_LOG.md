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
