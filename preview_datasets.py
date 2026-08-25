"""Izreže iz svakog dataseta jedan komad veličine dvorišta (40×40 m) i
spremi kao PLY — za vizualnu usporedbu s iPhone snimkom.

Pokreni na RunPod-u (gdje su sirovi dataseti):
    python preview_datasets.py --out /workspace/previews

Zatim skini PLY-eve lokalno i otvori u MeshLabu jedan po jedan.
"""
import argparse
import numpy as np
from pathlib import Path

import train as T   # koristi postojeće loadere


def crop_and_save(scan, out_path, size=40.0, max_pts=1_500_000):
    xyz = scan.xyz
    rgb = scan.rgb
    # središnji komad size×size m
    c = np.median(xyz[:, :2], axis=0)
    h = size / 2
    m = ((np.abs(xyz[:, 0] - c[0]) < h) & (np.abs(xyz[:, 1] - c[1]) < h))
    if m.sum() < 5000:                      # scan manji od izreza
        m = np.ones(len(xyz), dtype=bool)
    x = xyz[m]
    if len(x) > max_pts:
        pick = np.random.default_rng(0).choice(len(x), max_pts, replace=False)
        x = x[pick]
        col = rgb[m][pick] if rgb is not None else None
    else:
        col = rgb[m] if rgb is not None else None
    x = x - x.mean(axis=0, keepdims=True)

    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(x.astype(np.float64))
    if col is not None:
        c2 = col.astype(np.float64)
        if c2.max() > 1.5:
            c2 = c2 / 255.0
        p.colors = o3d.utility.Vector3dVector(np.clip(c2[:, :3], 0, 1))
    o3d.io.write_point_cloud(str(out_path), p)
    print(f"    → {out_path.name}: {len(x):,} točaka, "
          f"{'S BOJOM' if col is not None else 'BEZ BOJE'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/previews")
    ap.add_argument("--data", default="/workspace/data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    D = Path(args.data)

    jobs = [
        ("hessigheim", lambda: T.load_hessigheim(str(D / "Hessigheim3D"), split="val")),
        ("semantic3d", lambda: T.load_semantic3d(str(D / "Semantic3D"), stride=14)),
        ("parislille", lambda: T.load_parislille(str(D / "ParisLille3D"), stride=4)),
        ("toronto3d", lambda: T.load_toronto3d(str(D / "Toronto_3D"))),
        ("semantickitti", lambda: T.load_semantickitti(
            str(D / "SemanticKITTI"), train_sequences=["08"], stride=200)),
    ]
    for name, fn in jobs:
        print(f"\n=== {name} ===")
        try:
            scans = fn()
            if not scans:
                print("    (nema scanova)")
                continue
            crop_and_save(scans[0], out / f"preview_{name}.ply")
        except Exception as e:
            print(f"    [WARN] {e}")

    print(f"\nGotovo. Skini PLY-eve iz {out} i usporedi u MeshLabu.")


if __name__ == "__main__":
    main()
