#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

INPUT_ROOT = Path("spectrograms")
OUTPUT_ROOT = Path("dataset_split")

CLASSES = ["autel", "m30t", "background"]

SPLITS = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15,
}

random.seed(42)

def main():
    # crear carpetas
    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        in_dir = INPUT_ROOT / cls
        imgs = sorted(list(in_dir.glob("*.png")))
        if not imgs:
            print(f"[WARN] no images for {cls}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        n_test = n - n_train - n_val

        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train+n_val],
            "test":  imgs[n_train+n_val:],
        }

        print(f"{cls}: {n} -> "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")

        for split, files in splits.items():
            for src in files:
                dst = OUTPUT_ROOT / split / cls / src.name
                shutil.copy2(src, dst)

if __name__ == "__main__":
    main()
