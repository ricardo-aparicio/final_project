#!/usr/bin/env python3
import numpy as np
import pandas as pd   # si no lo tienes:  pip install pandas
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent

# ---------- DRNN ----------
drnn_hist_file = ROOT / "drnn_history_20ep.npz"
h = np.load(drnn_hist_file)

drnn_train_loss = h["train_loss"]
drnn_val_loss   = h["val_loss"]
drnn_train_acc  = h["train_acc"]
drnn_val_acc    = h["val_acc"]
epochs_drnn = np.arange(1, len(drnn_train_loss) + 1)

# ---------- YOLOv11 ----------
yolo_dir = ROOT / "runs_yolo11_cls" / "drone_spectrograms_20ep"
yolo_csv = yolo_dir / "results.csv"

df = pd.read_csv(yolo_csv)

# Ajusta los nombres si en tu CSV aparecen ligeramente diferentes
yolo_train_loss = df["train/loss"].values
yolo_val_loss   = df["val/loss"].values
yolo_val_acc    = df["metrics/accuracy_top1"].values   # top-1 (igual concepto que tu acc)
epochs_yolo = df["epoch"].values + 1  # YOLO suele numerar desde 0

# ---------- GRÁFICA: Loss ----------
plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_drnn, drnn_train_loss, "b-o", label="DRNN train")
plt.plot(epochs_drnn, drnn_val_loss,   "b--o", label="DRNN val")
plt.plot(epochs_yolo, yolo_train_loss, "r-s", label="YOLOv11 train")
plt.plot(epochs_yolo, yolo_val_loss,   "r--s", label="YOLOv11 val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch (DRNN vs YOLOv11)")
plt.grid(alpha=0.3)
plt.legend()

# ---------- GRÁFICA: Accuracy ----------
plt.subplot(1, 2, 2)
plt.plot(epochs_drnn, drnn_train_acc, "b-o", label="DRNN train acc")
plt.plot(epochs_drnn, drnn_val_acc,   "b--o", label="DRNN val acc")
plt.plot(epochs_yolo, yolo_val_acc,   "r-s", label="YOLOv11 val top-1")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch (DRNN vs YOLOv11)")
plt.ylim(0.97, 1.005)  # porque todo está muy alto
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
out_file = ROOT / "drnn_vs_yolo_training_curves.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"Comparación guardada en: {out_file}")
