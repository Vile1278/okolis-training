"""Dovrši cache: samo ParisLille + zapiši manifest (preskače sve ostalo).

Koristi se kad je cache skoro gotov a manifest ne postoji — izbjegava
ponovno učitavanje svih sirovih dataseta (~40 min).

Pokreni: python finish_cache.py
"""
import json
import shutil
from pathlib import Path
from train import load_parislille, preprocess_to_cache

CACHE = Path("/workspace/runs/ptv3_v2_fixed/scan_cache")

# KLJUČNO: kopiraj PLY-eve s network volumea na LOKALNI disk containera.
# plyfile čita memory-mapped (milijuni malih random pristupa) što je na
# network volumeu 10-50x sporije — zato je učitavanje "visjelo" satima.
# Sekvencijalna kopija je brza, čitanje s lokalnog diska još brže.
PL_LOCAL = Path("/tmp/ParisLille3D")
PL_LOCAL.mkdir(exist_ok=True)
print("=== 0/3: Kopiram PLY-eve na lokalni disk (brže čitanje) ===")
for f in sorted(Path("/workspace/data/ParisLille3D").glob("*.ply")):
    dst = PL_LOCAL / f.name
    if not dst.exists() or dst.stat().st_size != f.stat().st_size:
        print(f"  {f.name} ({f.stat().st_size/1e9:.1f} GB)...")
        shutil.copy(f, dst)
PL_ROOT = str(PL_LOCAL)

# Poznati raspored postojećeg cachea (iz redoslijeda u train.py):
# train: toronto 0-2, kitti 3-967, hessigheim 968, semantic3d 969-979
# val:   toronto 0, kitti 1-2, hessigheim 3, semantic3d 4-6
LAYOUT = [
    ("toronto3d", 0, 3),
    ("semantickitti", 3, 968),
    ("hessigheim", 968, 969),
    ("semantic3d", 969, 980),
    ("parislille", 980, 983),
]
N_VAL = 8  # 7 postojećih + 1 parislille

print("=== 1/3: ParisLille load + cache (3 train @ 980-982, 1 val @ 7) ===")
scans = load_parislille(PL_ROOT)
assert len(scans) == 4, f"Očekivana 4 scana, dobio {len(scans)}"
split = int(len(scans) * 0.8)  # 3
n_t = preprocess_to_cache(scans[:split], CACHE / "train", start_idx=980)
n_v = preprocess_to_cache(scans[split:], CACHE / "val", start_idx=7)
print(f"  cached: {n_t} train, {n_v} val")

print("=== 2/3: Provjera kompletnosti cachea ===")
ds_names = [name for name, _, _ in LAYOUT]
train_paths, train_ds_ids = [], []
for name, a, b in LAYOUT:
    for i in range(a, b):
        p = CACHE / "train" / f"scan_{i:06d}.npz"
        assert p.exists(), f"NEDOSTAJE {p} — raspored se ne poklapa!"
        train_paths.append(str(p))
        train_ds_ids.append(ds_names.index(name))
val_paths = []
for i in range(N_VAL):
    p = CACHE / "val" / f"scan_{i:06d}.npz"
    assert p.exists(), f"NEDOSTAJE {p}"
    val_paths.append(str(p))
print(f"  OK: {len(train_paths)} train + {len(val_paths)} val")

print("=== 3/3: Zapis manifesta ===")
(CACHE / "manifest.json").write_text(json.dumps({
    "datasets": sorted(ds_names),
    "ds_names": ds_names,
    "train_paths": train_paths,
    "train_ds_ids": train_ds_ids,
    "val_paths": val_paths,
}))
print(f"  MANIFEST spremljen: {CACHE / 'manifest.json'}")
print("\nGotovo! Sad pokreni trening — kreće za par sekundi:")
print("  nohup python -u train.py --config config.yaml > train.log 2>&1 &")
