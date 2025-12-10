#!/usr/bin/env python3
from pathlib import Path
import shutil

BASE = Path(".")  # carpeta actual: DETECTIONV1/dataset
OUT = BASE / "spectrograms"

# Define qué carpetas van a cada clase
AUTEL_FOLDERS = [
    "autel_con_rc02_30db", #1499 images in anechoic chamber
    "autel_sin_rc02_30db", #656 images in anechoic chamber
    "autel_sin_rc03_30db", #843 images in anechoic chamber
]

M30T_FOLDERS = [
    "m30t_con_rc_10m_30db", #750 images
    "m30t_con_rc_20m_30db", #750 images
    "m30t_con_rc_auto_30db", #750 images 
    "m30t_sin_rc_30db", #750 images
]

BACKGROUND_FOLDERS = [
    "ruido_camara01_30db", #1499 images 
    "ruido_camara02_30db", #103 images
]

CLASS_MAP = {
    "autel": AUTEL_FOLDERS,
    "m30t": M30T_FOLDERS,
    "background": BACKGROUND_FOLDERS,
}

def main():
    for cls, folders in CLASS_MAP.items():
        dest = OUT / cls
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{cls}] -> {dest}")

        for folder_name in folders:
            src_dir = BASE / folder_name
            if not src_dir.is_dir():
                print(f"  [WARN] {src_dir} no existe, lo salto")
                continue

            for img_path in src_dir.glob("*.png"):
                new_name = f"{folder_name}_{img_path.name}"
                dst_path = dest / new_name
                shutil.copy2(img_path, dst_path)

            print(f"  Copiado {folder_name}")

if __name__ == "__main__":
    main()
