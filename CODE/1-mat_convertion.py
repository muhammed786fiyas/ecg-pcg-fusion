import os
import wfdb
import scipy.io.wavfile as wav
import scipy.io as sio
import numpy as np

# Input and output folders
input_dir = "data"
output_dir = "mat_files"

os.makedirs(output_dir, exist_ok=True)

# Loop through all .hea files (one per record)
for file in os.listdir(input_dir):
    if file.endswith(".hea"):
        record_name = file.replace(".hea", "")
        record_path = os.path.join(input_dir, record_name)

        print(f"Processing: {record_name}")

        # -------- ECG (.dat + .hea) --------
        ecg_record = wfdb.rdrecord(record_path)
        ecg = ecg_record.p_signal          # ECG signal
        fs_ecg = ecg_record.fs             # ECG sampling rate
        ecg_channels = ecg_record.sig_name # Channel names

        # -------- PCG (.wav) --------
        wav_path = os.path.join(input_dir, record_name + ".wav")
        fs_pcg, pcg = wav.read(wav_path)

        # Convert PCG to float and normalize
        if pcg.dtype != np.float32:
            pcg = pcg.astype(np.float32)
            pcg /= np.max(np.abs(pcg))

        # -------- Save to .mat --------
        mat_path = os.path.join(output_dir, record_name + ".mat")
        sio.savemat(mat_path, {
            "ecg": ecg,
            "pcg": pcg,
            "fs_ecg": fs_ecg,
            "fs_pcg": fs_pcg,
            "ecg_channels": ecg_channels
        })

print("✅ All records converted to .mat format")
