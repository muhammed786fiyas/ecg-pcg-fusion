"""Objective test of which scalogram configs carry the CWT boundary artifact.

Independent of file timestamps and of which process wrote them: an unpadded ECG
scalogram has outer columns far brighter than its interior. Padded ones do not.
"""
import os
import numpy as np

root = "data/processed/scalograms"
print(f"{'config':18s} {'ECG edge/int':>13s} {'ECG interior':>13s} {'verdict':>12s}")
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name, "ecg.uint8.npy")
    if not os.path.exists(path):
        print(f"{name:18s} {'(no memmap)':>13s}")
        continue
    mm = np.load(path, mmap_mode="r")
    rows = np.linspace(0, mm.shape[0] - 1, 12).astype(int)
    sample = np.stack([np.asarray(mm[r]) for r in rows]).astype(np.float32)
    col = sample.mean(axis=(0, 1))
    edge = np.concatenate([col[:22], col[-22:]]).mean()
    interior = col[22:-22].mean()
    ratio = float(edge / max(interior, 1e-6))
    verdict = "PADDED ok" if ratio < 1.5 else "UNPADDED **"
    print(f"{name:18s} {ratio:13.2f} {interior:13.1f} {verdict:>12s}")
