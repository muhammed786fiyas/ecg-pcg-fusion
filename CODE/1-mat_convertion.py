import os
import wfdb
import scipy.io as sio
import numpy as np

input_dir = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\PHYSIONET RAW DATA\training-a"
output_dir = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\MATLAB DATA"

os.makedirs(output_dir, exist_ok=True)

skipped_records = []

for file in os.listdir(input_dir):
    if file.endswith(".hea"):
        record_name = file.replace(".hea", "")
        record_path = os.path.join(input_dir, record_name)

        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        fs = record.fs
        channel_names = record.sig_name

        # --- check availability ---
        if "ECG" not in channel_names or "PCG" not in channel_names:
            skipped_records.append(record_name)
            print(f"Skipping {record_name}: channels = {channel_names}")
            continue

        ecg = signals[:, channel_names.index("ECG")]
        ecg = ecg.astype(np.float32)

        pcg = signals[:, channel_names.index("PCG")]

        pcg = pcg.astype(np.float32)
        pcg /= np.max(np.abs(pcg))

        sio.savemat(
            os.path.join(output_dir, record_name + ".mat"),
            {
                "ecg": ecg,
                "pcg": pcg,
                "fs": fs,
                "channels": channel_names,
            },
        )

print("Conversion finished")
print(f"Skipped {len(skipped_records)} records without ECG+PCG")

