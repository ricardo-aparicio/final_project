#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from skimage.transform import resize
from pathlib import Path

FS = 50e6  # Hz,
INPUT_FILE = "autel_noisy01.bin"
OUTPUT_DIR = Path("dataset/autel_noisy01")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


iq = np.fromfile(INPUT_FILE, dtype=np.complex64)
print("Total samples:", len(iq))
print("Duration (s):", len(iq) / FS)


WINDOW_SAMPLES = 200_000
STEP_SAMPLES = 100_000   

num_segments = (len(iq) - WINDOW_SAMPLES) // STEP_SAMPLES + 1
print("Number of segments:", num_segments)

idx = 0
for start in range(0, len(iq) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
    seg = iq[start:start + WINDOW_SAMPLES]


    f, t, Sxx = spectrogram(
        seg,
        fs=FS,
        window="hann",
        nperseg=1024,
        noverlap=512,
        scaling="density",
        mode="magnitude"
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-9)

    # 4) Redimensionar a 256x256
    spec_256 = resize(
        Sxx_norm,
        (256, 256),
        mode="reflect",
        anti_aliasing=True
    )

    # 5) Guardar imagen
    out_path = OUTPUT_DIR / f"bg_{idx:04d}.png"
    plt.imsave(out_path, spec_256, cmap="viridis")
    idx += 1

print("Saved", idx, "spectrograms in", OUTPUT_DIR)

