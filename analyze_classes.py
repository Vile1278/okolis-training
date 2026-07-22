"""Analiza distribucije klasa po datasetu iz scan cache-a.

Pokazuje koliko svaki dataset doprinosi svakoj klasi — otkriva zasto
su neke klase (ground, sidewalk, fence, vehicle) na IoU=0.

Usage:
    python analyze_classes.py --cache /workspace/runs/ptv3_4ds/scan_cache
"""
import argparse
import numpy as np
from pathlib import Path

CLASS_NAMES = [
    "unlabeled", "ground", "road", "sidewalk",
    "building", "fence", "vegetation", "vehicle",
]
NUM_CLASSES = 8

# Redoslijed cachiranja (iz training loga):
# train: 0-2 toronto3d, 3 hessigheim, 4-14 semantic3d, 15-17 parislille
TRAIN_RANGES = {
    "toronto3d":  range(0, 3),
    "hessigheim": range(3, 4),
    "semantic3d": range(4, 15),
    "parislille": range(15, 18),
}


def analyze(cache_dir, split="train", ranges=None):
    d = Path(cache_dir) / split
    files = sorted(d.glob("scan_*.npz"))
    if not files:
        print(f"Nema fajlova u {d}")
        return

    print(f"\n{'='*80}")
    print(f"SPLIT: {split}  ({len(files)} scanova)")
    print(f"{'='*80}")

    # Per-dataset counts
    if ranges:
        ds_counts = {name: np.zeros(NUM_CLASSES, dtype=np.int64)
                     for name in ranges}
    total = np.zeros(NUM_CLASSES, dtype=np.int64)

    for f in files:
        idx = int(f.stem.split("_")[1])
        labels = np.load(f)["labels"]
        counts = np.bincount(labels, minlength=NUM_CLASSES)[:NUM_CLASSES]
        total += counts
        if ranges:
            for name, rng in ranges.items():
                if idx in rng:
                    ds_counts[name] += counts
                    break

    # Print per-dataset table
    if ranges:
        header = f"{'Klasa':<12}" + "".join(f"{n:>14}" for n in ranges) + f"{'UKUPNO':>14}"
        print(header)
        print("-" * len(header))
        for ci in range(NUM_CLASSES):
            row = f"{CLASS_NAMES[ci]:<12}"
            for name in ranges:
                c = ds_counts[name][ci]
                pct = 100 * c / max(ds_counts[name].sum(), 1)
                row += f"{c/1e6:>9.1f}M{pct:>4.0f}%"
            tc = total[ci]
            tpct = 100 * tc / max(total.sum(), 1)
            row += f"{tc/1e6:>9.1f}M{tpct:>4.0f}%"
            print(row)
        print("-" * len(header))

        # Upozorenja
        print("\n⚠ PROBLEMI:")
        for ci in range(1, NUM_CLASSES):  # skip unlabeled
            tc = total[ci]
            tpct = 100 * tc / max(total.sum(), 1)
            if tc == 0:
                print(f"  {CLASS_NAMES[ci]}: NEMA NIJEDNOG primjera — model je ne moze nauciti!")
            elif tpct < 1.0:
                print(f"  {CLASS_NAMES[ci]}: samo {tpct:.2f}% tocaka ({tc/1e6:.1f}M) — premalo, IoU ce biti nizak")
        ok = [CLASS_NAMES[ci] for ci in range(1, NUM_CLASSES)
              if 100 * total[ci] / max(total.sum(), 1) >= 1.0]
        print(f"\n✓ Dovoljno primjera: {', '.join(ok)}")
    else:
        for ci in range(NUM_CLASSES):
            tc = total[ci]
            tpct = 100 * tc / max(total.sum(), 1)
            print(f"  {CLASS_NAMES[ci]:<12} {tc/1e6:>10.1f}M  {tpct:>5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/runs/ptv3_4ds/scan_cache")
    args = ap.parse_args()
    analyze(args.cache, "train", TRAIN_RANGES)
    analyze(args.cache, "val", None)
