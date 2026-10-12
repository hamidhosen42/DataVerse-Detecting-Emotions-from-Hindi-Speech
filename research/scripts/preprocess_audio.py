"""Audio preprocessing variants to remove the PCM_16 / PCM_32 recording-batch differences.
V1: trim leading/trailing low-energy frames (< peak - 35 dB, 20 ms frames, keep 50 ms margin) + RMS-normalise to -20 dBFS.
V2: V1 + 4 kHz low-pass (8th-order Butterworth, zero-phase) applied to every clip.
Usage: python -I preprocess_audio.py <comp_paths.csv> <out_v1_dir> <out_v2_dir> <out_list_v1.csv> <out_list_v2.csv>
"""
import os, sys
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import butter, sosfiltfilt
lst, o1, o2, l1, l2 = sys.argv[1:6]
sos = butter(8, 4000, btype="low", fs=16000, output="sos")
p1, p2, cut = [], [], []
for p in pd.read_csv(lst).path:
    y, sr = sf.read(p, dtype="float32")
    fr = 320
    e = np.array([np.sqrt(np.mean(y[i:i + fr] ** 2) + 1e-12) for i in range(0, len(y), fr)])
    db = 20 * np.log10(e + 1e-12); keep = np.where(db > db.max() - 35)[0]
    s, t = max(0, keep[0] * fr - 800), min(len(y), (keep[-1] + 1) * fr + 800)
    z = y[s:t]
    if len(z) < 8000: z = y            # never trim below 0.5 s
    z = z * (10 ** (-20 / 20) / (np.sqrt(np.mean(z ** 2)) + 1e-9))
    z = np.clip(z, -0.99, 0.99).astype(np.float32)
    f1 = f"{o1}/{os.path.basename(p)}"; sf.write(f1, z, 16000); p1.append(f1)
    zz = np.clip(sosfiltfilt(sos, z), -0.99, 0.99).astype(np.float32)
    f2 = f"{o2}/{os.path.basename(p)}"; sf.write(f2, zz, 16000); p2.append(f2)
    cut.append(1 - len(z) / len(y))
pd.DataFrame({"path": p1}).to_csv(l1, index=False); pd.DataFrame({"path": p2}).to_csv(l2, index=False)
print("done", len(p1), "| mean fraction trimmed", round(float(np.mean(cut)), 3))
