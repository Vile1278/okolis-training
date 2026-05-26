"""Standalone training script for Okolis AI.

Trains Point Transformer V3 on Toronto3D + SemanticKITTI + Pandaset + 3DRef
for 8-class outdoor segmentation.  All helpers are inline.

Usage:
    python train.py --config config.yaml
"""
import argparse
import math
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from model import PointTransformerV3, RandLANet  # RandLANet is alias
from losses import LovaszSoftmax, class_weighted_ce


# ============================================================================
# Label Maps (8 classes)
# ============================================================================
NUM_CLASSES = 8
CLASS_NAMES = [
    "unlabeled", "ground", "road", "sidewalk",
    "building", "fence", "vegetation", "vehicle",
]

# Toronto3D: 0=unclass, 1=ground, 2=road_marking, 3=natural(veg),
#             4=building, 5=utility_line, 6=pole, 7=car, 8=fence
TORONTO3D_MAP = {
    0: 0, 1: 2, 2: 2, 3: 6, 4: 4, 5: 0, 6: 0, 7: 7, 8: 5,
    #                                 ^pole→unlabeled  ^car→vehicle
}

# SemanticKITTI RAW label -> unified 8-class
SEMKITTI_RAW_MAP = {
    0: 0, 1: 0,
    10: 7, 11: 7, 13: 7, 15: 7, 16: 7, 18: 7, 20: 7,  # car/truck/motorcycle/etc → vehicle
    30: 0, 31: 0, 32: 0,                                  # persons → unlabeled
    40: 2, 44: 3, 48: 3, 49: 1,                            # road/parking/sidewalk/other-ground
    50: 4, 51: 5, 52: 0,                                   # building/fence/other-struct
    60: 2,                                                  # lane-marking → road
    70: 6, 71: 6, 72: 1,                                   # vegetation/trunk/terrain
    80: 0, 81: 0,                                           # pole/traffic-sign → unlabeled
    99: 0,
    252: 7, 253: 7, 254: 7, 255: 7, 256: 7, 257: 7, 258: 7, 259: 7,  # moving vehicles → vehicle
}

# Pandaset class mapping → unified 8-class (from classes.json)
# 1=Smoke, 2=Exhaust, 3=Spray/rain, 4=Reflection, 5=Vegetation,
# 6=Ground, 7=Road, 8=Lane Line, 9=Stop Line, 10=Other Road Marking,
# 11=Sidewalk, 12=Driveway, 13=Car, 14=Pickup Truck, 15=Medium Truck,
# 16=Semi-truck, 17=Towed Object, 18=Motorcycle, 19=Construction Vehicle,
# 20=Uncommon Vehicle, 21=Pedicab, 22=Emergency Vehicle, 23=Bus,
# 24=Personal Mobility, 25=Motorized Scooter, 26=Bicycle, 27=Train,
# 28=Trolley, 29=Tram, 30=Pedestrian, 31=Pedestrian+Object,
# 32=Bird, 33=Animals, 34=Pylons, 35=Road Barriers, 36=Signs,
# 37=Cones, 38=Construction Signs, 39=Temp Barriers, 40=Rolling Containers,
# 41=Building, 42=Other Static Object
PANDASET_MAP = {
    1: 0, 2: 0, 3: 0, 4: 0,             # smoke/exhaust/spray/reflection → unlabeled
    5: 6,    # Vegetation → vegetation
    6: 1,    # Ground → ground
    7: 2,    # Road → road
    8: 2,    # Lane Line Marking → road
    9: 2,    # Stop Line Marking → road
    10: 2,   # Other Road Marking → road
    11: 3,   # Sidewalk → sidewalk
    12: 3,   # Driveway → sidewalk
    13: 7,   # Car → vehicle
    14: 7,   # Pickup Truck → vehicle
    15: 7,   # Medium Truck → vehicle
    16: 7,   # Semi-truck → vehicle
    17: 7,   # Towed Object → vehicle
    18: 7,   # Motorcycle → vehicle
    19: 7,   # Construction Vehicle → vehicle
    20: 7,   # Uncommon Vehicle → vehicle
    21: 7,   # Pedicab → vehicle
    22: 7,   # Emergency Vehicle → vehicle
    23: 7,   # Bus → vehicle
    24: 0, 25: 0, 26: 0,                  # personal mobility/scooter/bicycle → unlabeled
    27: 0, 28: 0, 29: 0,                  # train/trolley/tram → unlabeled
    30: 0, 31: 0,                          # pedestrians → unlabeled
    32: 0, 33: 0,                          # animals → unlabeled
    34: 0, 35: 5, 36: 0,                  # pylons→unlabeled, road barriers→fence, signs→unlabeled
    37: 0, 38: 0, 39: 5,                  # cones→unlabeled, construction signs→unlabeled, temp barriers→fence
    40: 0,                                 # rolling containers → unlabeled
    41: 4,   # Building → building
    42: 0,   # Other Static Object → unlabeled
}

# 3DRef (Livox Avia, solid-state LiDAR closest to iPhone)
# NOTE: 3DRef is mixed indoor/outdoor. We keep outdoor-relevant classes.
THREEREF_MAP = {
    0: 0,    # unlabeled
    1: 4,    # wall → building
    2: 1,    # floor → ground
    3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
    11: 0, 12: 0, 13: 0, 14: 5, 15: 0, 16: 0, 17: 0,
    18: 0, 19: 0, 20: 0, 21: 0,
    # Extended outdoor labels (if present in 3DRef outdoor subset)
    100: 1,  # ground
    101: 2,  # road
    102: 3,  # sidewalk
    103: 4,  # building
    104: 5,  # fence
    105: 6,  # vegetation
    106: 7,  # vehicle
}


# Hessigheim 3D (UAV LiDAR, German village — residential!)
# Classes: 0=Low Vegetation, 1=Impervious Surface, 2=Vehicle,
#          3=Urban Furniture, 4=Roof, 5=Façade, 6=Shrub,
#          7=Tree, 8=Soil/Gravel, 9=Vertical Surface, 10=Chimney
HESSIGHEIM_MAP = {
    0: 6,    # Low Vegetation → vegetation
    1: 3,    # Impervious Surface → sidewalk (paved ground)
    2: 7,    # Vehicle → vehicle
    3: 0,    # Urban Furniture → unlabeled
    4: 4,    # Roof → building
    5: 4,    # Façade → building
    6: 6,    # Shrub → vegetation
    7: 6,    # Tree → vegetation
    8: 1,    # Soil/Gravel → ground
    9: 4,    # Vertical Surface → building (walls etc.)
    10: 4,   # Chimney → building
}

# Semantic3D (terrestrial scanner, European villages/towns)
# Classes: 1=man-made terrain, 2=natural terrain, 3=high vegetation,
#          4=low vegetation, 5=buildings, 6=hard scape, 7=scanning artefacts, 8=cars
SEMANTIC3D_MAP = {
    0: 0,    # unlabeled
    1: 2,    # man-made terrain → road (concrete/asphalt surfaces)
    2: 1,    # natural terrain → ground
    3: 6,    # high vegetation → vegetation
    4: 6,    # low vegetation → vegetation
    5: 4,    # buildings → building
    6: 3,    # hard scape (sidewalks, curbs) → sidewalk
    7: 0,    # scanning artefacts → unlabeled
    8: 7,    # cars → vehicle
}

# DALES (aerial LiDAR, urban+suburban USA — has fence class!)
# Classes: 1=ground, 2=vegetation, 3=cars, 4=trucks, 5=power lines,
#          6=fences, 7=poles, 8=buildings
DALES_MAP = {
    0: 0,    # unlabeled
    1: 1,    # ground → ground
    2: 6,    # vegetation → vegetation
    3: 7,    # cars → vehicle
    4: 7,    # trucks → vehicle
    5: 0,    # power lines → unlabeled
    6: 5,    # fences → fence
    7: 0,    # poles → unlabeled
    8: 4,    # buildings → building
}

# ISPRS Vaihingen 3D (aerial LiDAR, German suburb — residential!)
# Classes: 1=Powerline, 2=Low Vegetation, 3=Impervious Surface,
#          4=Car, 5=Fence/Hedge, 6=Roof, 7=Façade, 8=Shrub, 9=Tree
VAIHINGEN_MAP = {
    0: 0,    # unlabeled
    1: 0,    # Powerline → unlabeled
    2: 6,    # Low Vegetation → vegetation (grass, lawn)
    3: 3,    # Impervious Surface → sidewalk (paved ground)
    4: 7,    # Car → vehicle
    5: 5,    # Fence/Hedge → fence
    6: 4,    # Roof → building
    7: 4,    # Façade → building
    8: 6,    # Shrub → vegetation
    9: 6,    # Tree → vegetation
}

# SensatUrban (UAV photogrammetry, UK cities)
# Classes: 0=ground, 1=high vegetation, 2=buildings, 3=walls,
#          4=bridge, 5=parking, 6=rail, 7=traffic roads, 8=street furniture,
#          9=cars, 10=footpath, 11=bikes, 12=water
SENSATURBAN_MAP = {
    0: 1,    # ground → ground
    1: 6,    # high vegetation → vegetation
    2: 4,    # buildings → building
    3: 4,    # walls → building
    4: 0,    # bridge → unlabeled
    5: 2,    # parking → road
    6: 0,    # rail → unlabeled
    7: 2,    # traffic roads → road
    8: 0,    # street furniture → unlabeled
    9: 7,    # cars → vehicle
    10: 3,   # footpath → sidewalk
    11: 0,   # bikes → unlabeled
    12: 0,   # water → unlabeled
}

# Paris-Lille-3D (mobile LiDAR, French cities — street level!)
# Coarse classes: 0=unclassified, 1=ground, 2=building, 3=pole,
#                 4=bollard, 5=trash can, 6=barrier, 7=pedestrian,
#                 8=car, 9=natural (vegetation)
PARISLILLE_MAP = {
    0: 0,    # unclassified → unlabeled
    1: 2,    # ground → road (street-level MLS on roads)
    2: 4,    # building → building
    3: 0,    # pole → unlabeled
    4: 0,    # bollard → unlabeled
    5: 0,    # trash can → unlabeled
    6: 5,    # barrier → fence
    7: 0,    # pedestrian → unlabeled
    8: 7,    # car → vehicle
    9: 6,    # natural → vegetation
}


def apply_map(labels, mapping):
    """Remap integer labels via a dict."""
    out = np.zeros_like(labels, dtype=np.int64)
    for src, dst in mapping.items():
        out[labels == src] = dst
    return out


# ============================================================================
# Feature Utilities
# ============================================================================
FEAT_DIM = 5


def pack_features(rgb, intensity, hag):
    """Pack into (N, 5): [R, G, B, intensity, height_above_ground]."""
    N = len(hag)
    feats = np.zeros((N, FEAT_DIM), dtype=np.float32)
    if rgb is not None:
        feats[:, 0:3] = rgb[:, :3]
    if intensity is not None:
        feats[:, 3] = intensity
    feats[:, 4] = hag
    return feats


def height_above_ground_from_labels(xyz, labels, ground_label=1, cell=1.0):
    """Compute per-point height above ground using a gridded ground surface."""
    from scipy.ndimage import median_filter

    ground_mask = labels == ground_label
    if ground_mask.sum() < 10:
        z_thresh = np.percentile(xyz[:, 2], 10)
        ground_mask = xyz[:, 2] <= z_thresh

    gx = xyz[ground_mask, 0]
    gy = xyz[ground_mask, 1]
    gz = xyz[ground_mask, 2]

    xmin, ymin = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    xmax, ymax = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))

    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    if len(gx) > 0:
        ci = np.clip(((gx - xmin) / cell).astype(int), 0, nx - 1)
        cj = np.clip(((gy - ymin) / cell).astype(int), 0, ny - 1)
        np.minimum.at(grid, (ci, cj), gz.astype(np.float32))
    grid[np.isinf(grid)] = np.nan

    # Fill NaN cells
    for _ in range(5):
        if not np.any(np.isnan(grid)):
            break
        med = np.nanmedian(grid) if not np.all(np.isnan(grid)) else np.median(xyz[:, 2])
        filled = median_filter(np.nan_to_num(grid, nan=med), size=3)
        mask = np.isnan(grid)
        grid[mask] = filled[mask]

    if np.any(np.isnan(grid)):
        grid[np.isnan(grid)] = np.nanmedian(grid)

    pi = np.clip(((xyz[:, 0] - xmin) / cell).astype(int), 0, nx - 1)
    pj = np.clip(((xyz[:, 1] - ymin) / cell).astype(int), 0, ny - 1)
    ground_z = grid[pi, pj]
    hag = xyz[:, 2] - ground_z
    return np.clip(hag, 0.0, None).astype(np.float32)


def modality_dropout(feats, drop_rgb_p=0.3, drop_intensity_p=0.3):
    """Randomly zero out RGB and/or intensity during training."""
    feats = feats.copy()
    if np.random.rand() < drop_rgb_p:
        feats[:, 0:3] = 0.0
    if np.random.rand() < drop_intensity_p:
        feats[:, 3] = 0.0
    return feats


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class LoadedScan:
    xyz: np.ndarray
    rgb: Optional[np.ndarray]
    intensity: Optional[np.ndarray]
    labels: np.ndarray


def load_toronto3d(root):
    """Load Toronto3D PLY files. Returns list of LoadedScan."""
    from plyfile import PlyData

    scans = []
    ply_dir = Path(root)
    ply_files = sorted(ply_dir.glob("L00*.ply"))
    if not ply_files:
        # Try globbing all .ply
        ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        print(f"  [WARN] No .ply files in {root}")
        return scans

    for f in ply_files:
        print(f"  Loading Toronto3D: {f.name} ...", end=" ", flush=True)
        ply = PlyData.read(str(f))
        v = ply["vertex"].data
        N = len(v)
        print(f"{N:,} points")

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        rgb = None
        if all(k in v.dtype.names for k in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]],
                           axis=1).astype(np.float32) / 255.0

        intensity = None
        for key in ("scalar_Intensity", "intensity", "Intensity"):
            if key in v.dtype.names:
                raw = np.asarray(v[key], dtype=np.float32)
                mx = max(float(raw.max()), 1.0)
                intensity = (raw / mx).astype(np.float32)
                break

        label_raw = None
        for key in ("scalar_Label", "label", "class"):
            if key in v.dtype.names:
                label_raw = np.asarray(v[key], dtype=np.int64)
                break
        if label_raw is None:
            print(f"    [WARN] No label field in {f.name}, skipping")
            continue

        labels = apply_map(label_raw, TORONTO3D_MAP)
        del ply, v
        scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity, labels=labels))

    total_pts = sum(s.xyz.shape[0] for s in scans)
    print(f"  Toronto3D total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_semantickitti(root, train_sequences=None, stride=1):
    """Load SemanticKITTI .bin + .label files. Returns list of LoadedScan.

    Args:
        stride: load every Nth scan to reduce RAM usage.  stride=5 cuts
                 ~19K training scans to ~3.8K, reducing RAM from ~92 GB
                 to ~18 GB.  The model still sees diverse scenes because
                 consecutive scans overlap heavily.
    """
    if train_sequences is None:
        train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]

    base = Path(root) / "dataset" / "sequences"
    if not base.exists():
        base = Path(root) / "sequences"
    if not base.exists():
        print(f"  [WARN] SemanticKITTI sequences not found at {root}")
        return []

    scans = []
    for seq in train_sequences:
        vel_dir = base / seq / "velodyne"
        lbl_dir = base / seq / "labels"
        if not vel_dir.exists() or not lbl_dir.exists():
            print(f"  [WARN] Missing seq {seq}")
            continue

        bin_files = sorted(vel_dir.glob("*.bin"))[::stride]
        print(f"  Loading SemanticKITTI seq {seq}: {len(bin_files)} scans (stride={stride})")

        for bf in bin_files:
            lf = lbl_dir / bf.name.replace(".bin", ".label")
            if not lf.exists():
                continue

            pts = np.fromfile(str(bf), dtype=np.float32).reshape(-1, 4)
            xyz = pts[:, :3].astype(np.float32)
            intensity = np.clip(pts[:, 3], 0.0, 1.0).astype(np.float32)

            raw = np.fromfile(str(lf), dtype=np.uint32)
            sem_labels = (raw & 0xFFFF).astype(np.int64)
            labels = apply_map(sem_labels, SEMKITTI_RAW_MAP)

            scans.append(LoadedScan(xyz=xyz, rgb=None, intensity=intensity, labels=labels))

    print(f"  SemanticKITTI total: {len(scans)} scans")
    return scans


def load_pandaset(root, stride=1):
    """Load Pandaset LiDAR sequences. Returns list of LoadedScan.

    Pandaset structure (Kaggle version):
        root/001/lidar/00.pkl, 01.pkl, ...
        root/001/annotations/semseg/00.pkl, 01.pkl, ...

    Each .pkl is a pickled pandas DataFrame.
    Lidar: columns x, y, z, i, t, d
    Semseg: single column with integer class labels.

    Args:
        root: path to pandaset root (containing sequence folders like 001, 002...)
        stride: load every Nth frame
    """
    import pickle

    base = Path(root)
    scans = []

    seq_dirs = sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith('.')])
    print(f"  Pandaset: {len(seq_dirs)} sequences")

    for seq_dir in seq_dirs:
        lidar_dir = seq_dir / "lidar"
        labels_dir = seq_dir / "annotations" / "semseg"
        if not lidar_dir.exists() or not labels_dir.exists():
            continue

        # List .pkl files directly in lidar/ folder
        pkl_files = sorted(lidar_dir.glob("*.pkl"))[::stride]
        loaded = 0

        for pkl_path in pkl_files:
            frame_name = pkl_path.stem  # e.g. "00", "01"
            lbl_path = labels_dir / f"{frame_name}.pkl"

            if not lbl_path.exists():
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    df = pickle.load(f)
                xyz = df[['x', 'y', 'z']].values.astype(np.float32)
                intensity = None
                if 'i' in df.columns:
                    raw_i = df['i'].values.astype(np.float32)
                    mx = max(float(np.nanmax(raw_i)), 1.0)
                    intensity = np.clip(raw_i / mx, 0, 1).astype(np.float32)

                with open(lbl_path, 'rb') as f:
                    lbl_df = pickle.load(f)
                label_raw = lbl_df.values.flatten().astype(np.int64)

                if len(xyz) != len(label_raw):
                    continue

                labels = apply_map(label_raw, PANDASET_MAP)
                scans.append(LoadedScan(xyz=xyz, rgb=None, intensity=intensity, labels=labels))
                loaded += 1
            except Exception as e:
                print(f"    [WARN] Failed {pkl_path}: {e}")
                continue

        if loaded > 0:
            print(f"    Seq {seq_dir.name}: {loaded} frames")

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  Pandaset total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_3dref(root, stride=1):
    """Load 3DRef dataset (Livox Avia solid-state LiDAR).

    3DRef stores scans as .ply or .pcd files with semantic labels.
    The scan pattern (non-repetitive) is closest to iPhone LiDAR.

    Args:
        root: path to 3DRef root
        stride: load every Nth scan
    """
    base = Path(root)
    scans = []

    # Try different directory structures
    scan_dirs = []
    for pattern in ["*.ply", "*.pcd", "**/*.ply", "**/*.pcd",
                     "scans/*.ply", "point_clouds/*.ply"]:
        found = sorted(base.glob(pattern))
        if found:
            scan_dirs = found[::stride]
            break

    # Also try .npy format
    if not scan_dirs:
        npy_files = sorted(base.glob("**/*.npy"))
        if npy_files:
            scan_dirs = npy_files[::stride]

    print(f"  3DRef: {len(scan_dirs)} scan files")

    for f in scan_dirs:
        try:
            if f.suffix == ".ply":
                from plyfile import PlyData
                ply = PlyData.read(str(f))
                v = ply["vertex"].data
                xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

                # Look for labels
                label_raw = None
                for key in ("label", "class", "scalar_Label", "semantic"):
                    if key in v.dtype.names:
                        label_raw = np.asarray(v[key], dtype=np.int64)
                        break

                rgb = None
                if all(k in v.dtype.names for k in ("red", "green", "blue")):
                    rgb = np.stack([v["red"], v["green"], v["blue"]],
                                   axis=1).astype(np.float32)
                    if rgb.max() > 1.0:
                        rgb /= 255.0

                intensity = None
                for key in ("intensity", "i", "scalar_Intensity"):
                    if key in v.dtype.names:
                        raw = np.asarray(v[key], dtype=np.float32)
                        mx = max(float(raw.max()), 1.0)
                        intensity = (raw / mx).astype(np.float32)
                        break

                if label_raw is None:
                    # Try companion .labels or .txt file
                    lbl_file = f.with_suffix(".labels")
                    if not lbl_file.exists():
                        lbl_file = f.with_suffix(".txt")
                    if lbl_file.exists():
                        label_raw = np.loadtxt(str(lbl_file), dtype=np.int64)

                if label_raw is None or len(xyz) == 0:
                    continue

                labels = apply_map(label_raw, THREEREF_MAP)
                scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                        labels=labels))

            elif f.suffix == ".npy":
                data = np.load(f)
                if data.shape[1] < 4:
                    continue
                xyz = data[:, :3].astype(np.float32)
                label_raw = data[:, -1].astype(np.int64)
                rgb = None
                intensity = None
                if data.shape[1] >= 7:  # x,y,z,r,g,b,label
                    rgb = data[:, 3:6].astype(np.float32)
                    if rgb.max() > 1.0:
                        rgb /= 255.0

                labels = apply_map(label_raw, THREEREF_MAP)
                scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                        labels=labels))
        except Exception as e:
            print(f"    [WARN] Failed to load {f}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  3DRef total: {len(scans)} scans, {total_pts:,} points")
    return scans


def _load_ply_generic(filepath):
    """Load a PLY file and return (vertex_data, xyz, rgb, intensity)."""
    from plyfile import PlyData
    ply = PlyData.read(str(filepath))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    rgb = None
    for r_key, g_key, b_key in [("red", "green", "blue"), ("r", "g", "b")]:
        if all(k in v.dtype.names for k in (r_key, g_key, b_key)):
            rgb = np.stack([v[r_key], v[g_key], v[b_key]],
                           axis=1).astype(np.float32)
            if rgb.max() > 1.0:
                rgb /= 255.0
            break

    intensity = None
    for key in ("intensity", "scalar_Intensity", "Intensity", "i",
                "reflectance", "scalar_Reflectance"):
        if key in v.dtype.names:
            raw = np.asarray(v[key], dtype=np.float32)
            mx = max(float(np.nanmax(raw)), 1.0)
            intensity = np.clip(raw / mx, 0, 1).astype(np.float32)
            break

    return v, xyz, rgb, intensity


def _load_laz(filepath):
    """Load a LAZ/LAS file. Returns (xyz, rgb, intensity, labels)."""
    import laspy
    las = laspy.read(str(filepath))
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)

    rgb = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.asarray(las.red, dtype=np.float32)
        g = np.asarray(las.green, dtype=np.float32)
        b = np.asarray(las.blue, dtype=np.float32)
        # LAS RGB is often 16-bit (0-65535)
        mx = max(float(r.max()), float(g.max()), float(b.max()), 1.0)
        if mx > 255:
            rgb = np.stack([r / 65535.0, g / 65535.0, b / 65535.0], axis=1)
        elif mx > 1:
            rgb = np.stack([r / 255.0, g / 255.0, b / 255.0], axis=1)
        else:
            rgb = np.stack([r, g, b], axis=1)
        rgb = rgb.astype(np.float32)

    intensity = None
    if hasattr(las, 'intensity'):
        raw_i = np.asarray(las.intensity, dtype=np.float32)
        mx = max(float(np.nanmax(raw_i)), 1.0)
        intensity = np.clip(raw_i / mx, 0, 1).astype(np.float32)

    labels = None
    if hasattr(las, 'classification'):
        labels = np.asarray(las.classification, dtype=np.int64)

    return xyz, rgb, intensity, labels


def load_hessigheim(root, split="all"):
    """Load Hessigheim 3D (UAV LiDAR, German village).

    H3D structure (after extraction):
        root/Mar19_train.laz   (or .txt)
        root/Mar19_val.laz
        root/Mar19_test_GroundTruth.laz

    Supports LAZ/LAS, PLY, and TXT formats.
    The dataset has pre-defined train/val/test splits.

    Args:
        root: path to Hessigheim3D folder
        split: "train", "val", "test", or "all"
    """
    base = Path(root)
    scans = []

    # Find files matching split pattern
    patterns = []
    if split in ("train", "all"):
        patterns.append("*train*")
    if split in ("val", "all"):
        patterns.append("*val*")
    if split in ("test", "all"):
        patterns.append("*test*GroundTruth*")  # test without GT has no labels

    scan_files = []
    for pat in patterns:
        for ext in (".laz", ".las", ".ply", ".txt"):
            found = sorted(base.glob(pat + ext))
            scan_files.extend(found)
        # Also search subdirectories (e.g., Epoch_March2019/LiDAR/)
        for ext in (".laz", ".las", ".ply"):
            found = sorted(base.glob("**/" + pat + ext))
            scan_files.extend(found)

    # Deduplicate
    scan_files = list(dict.fromkeys(scan_files))
    print(f"  Hessigheim ({split}): {len(scan_files)} files")

    for f in scan_files:
        try:
            if f.suffix in (".laz", ".las"):
                xyz, rgb, intensity, label_raw = _load_laz(f)
                print(f"    {f.name}: {len(xyz):,} points (LAZ)", end="")
            elif f.suffix == ".ply":
                v, xyz, rgb, intensity = _load_ply_generic(f)
                label_raw = None
                for key in ("label", "class", "classification",
                            "scalar_Label", "scalar_Classification"):
                    if key in v.dtype.names:
                        label_raw = np.asarray(v[key], dtype=np.int64)
                        break
                print(f"    {f.name}: {len(xyz):,} points (PLY)", end="")
            elif f.suffix == ".txt":
                # H3D TXT: x y z intensity r g b label (space separated)
                data = np.loadtxt(str(f), dtype=np.float32,
                                  max_rows=None, comments='/')
                xyz = data[:, :3].astype(np.float32)
                intensity = None
                rgb = None
                label_raw = None
                if data.shape[1] >= 8:
                    intensity = np.clip(data[:, 3] / max(data[:, 3].max(), 1), 0, 1).astype(np.float32)
                    rgb = data[:, 4:7].astype(np.float32)
                    if rgb.max() > 1.0:
                        rgb /= 255.0 if rgb.max() <= 255 else 65535.0
                    label_raw = data[:, 7].astype(np.int64)
                elif data.shape[1] >= 4:
                    label_raw = data[:, -1].astype(np.int64)
                print(f"    {f.name}: {len(xyz):,} points (TXT)", end="")
            else:
                continue

            if label_raw is None:
                print(" [SKIP: no labels]")
                continue

            labels = apply_map(label_raw, HESSIGHEIM_MAP)
            scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                    labels=labels))
            print(f" → mapped")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  Hessigheim total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_semantic3d(root, stride=1):
    """Load Semantic3D (terrestrial scanner, European villages).

    Semantic3D format:
        root/station1.txt          — space-separated: x y z intensity r g b
        root/station1.labels       — one label per line (integer)
    """
    base = Path(root)
    scans = []

    # Find .txt files that have matching .labels
    txt_files = sorted(base.glob("*.txt"))
    pairs = []
    for tf in txt_files:
        lf = tf.with_suffix(".labels")
        if lf.exists():
            pairs.append((tf, lf))
    pairs = pairs[::stride]
    print(f"  Semantic3D: {len(pairs)} scan+label pairs")

    for txt_path, lbl_path in pairs:
        try:
            print(f"    {txt_path.name}: ", end="", flush=True)
            # Semantic3D files can be very large — use numpy for speed
            data = np.loadtxt(str(txt_path), dtype=np.float32)
            # columns: x, y, z, intensity, r, g, b
            xyz = data[:, :3].astype(np.float32)

            intensity = None
            if data.shape[1] >= 4:
                raw_i = data[:, 3]
                mx = max(float(np.nanmax(raw_i)), 1.0)
                intensity = np.clip(raw_i / mx, 0, 1).astype(np.float32)

            rgb = None
            if data.shape[1] >= 7:
                rgb = data[:, 4:7].astype(np.float32)
                if rgb.max() > 1.0:
                    rgb /= 255.0

            label_raw = np.loadtxt(str(lbl_path), dtype=np.int64)
            if len(label_raw) != len(xyz):
                print(f"[SKIP: label count mismatch {len(label_raw)} vs {len(xyz)}]")
                continue

            labels = apply_map(label_raw, SEMANTIC3D_MAP)
            scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                    labels=labels))
            print(f"{len(xyz):,} points → mapped")
            del data
        except Exception as e:
            print(f"[WARN: {e}]")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  Semantic3D total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_dales(root, stride=1):
    """Load DALES (aerial LiDAR, urban+suburban USA).

    DALES provides .ply or .las files with per-point labels.
    Structure: root/train/*.ply and root/test/*.ply
    PLY fields: x, y, z, (+ possibly intensity, classification/label)
    """
    base = Path(root)
    scans = []

    scan_files = []
    for subdir in ["train", "test", "."]:
        d = base / subdir if subdir != "." else base
        for ext in ("*.ply", "*.las", "*.laz"):
            found = sorted(d.glob(ext))
            scan_files.extend(found)
    # Deduplicate
    scan_files = list(dict.fromkeys(scan_files))
    scan_files = scan_files[::stride]
    print(f"  DALES: {len(scan_files)} files")

    for f in scan_files:
        try:
            if f.suffix in (".las", ".laz"):
                # LAS/LAZ format using laspy
                import laspy
                las = laspy.read(str(f))
                xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
                intensity = None
                if hasattr(las, 'intensity'):
                    raw_i = np.asarray(las.intensity, dtype=np.float32)
                    mx = max(float(raw_i.max()), 1.0)
                    intensity = (raw_i / mx).astype(np.float32)
                label_raw = np.asarray(las.classification, dtype=np.int64)
                rgb = None
                labels = apply_map(label_raw, DALES_MAP)
                scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                        labels=labels))
                print(f"    {f.name}: {len(xyz):,} points (LAS)")
            else:
                v, xyz, rgb, intensity = _load_ply_generic(f)
                print(f"    {f.name}: {len(xyz):,} points", end="")

                label_raw = None
                for key in ("classification", "label", "class", "scalar_Label",
                            "scalar_Classification"):
                    if key in v.dtype.names:
                        label_raw = np.asarray(v[key], dtype=np.int64)
                        break
                if label_raw is None:
                    print(" [SKIP: no labels]")
                    continue

                labels = apply_map(label_raw, DALES_MAP)
                scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                        labels=labels))
                print(f" → mapped")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  DALES total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_vaihingen(root, stride=1):
    """Load ISPRS Vaihingen 3D (aerial LiDAR, German suburb).

    Structure: root/*.ply with per-point labels
    9 classes: powerline, low_veg, impervious, car, fence, roof, facade, shrub, tree
    """
    base = Path(root)
    scans = []

    ply_files = sorted(base.glob("*.ply"))
    if not ply_files:
        ply_files = sorted(base.glob("**/*.ply"))
    ply_files = ply_files[::stride]
    print(f"  Vaihingen: {len(ply_files)} files")

    for f in ply_files:
        try:
            v, xyz, rgb, intensity = _load_ply_generic(f)
            print(f"    {f.name}: {len(xyz):,} points", end="")

            label_raw = None
            for key in ("label", "class", "classification", "scalar_Label"):
                if key in v.dtype.names:
                    label_raw = np.asarray(v[key], dtype=np.int64)
                    break
            if label_raw is None:
                print(" [SKIP: no labels]")
                continue

            labels = apply_map(label_raw, VAIHINGEN_MAP)
            scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                    labels=labels))
            print(f" → mapped")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  Vaihingen total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_sensaturban(root, stride=1):
    """Load SensatUrban (UAV photogrammetry, UK cities).

    Structure: root/train/*.ply and root/test/*.ply
    Each PLY: x, y, z, red, green, blue, label
    13 classes (0-12).
    """
    base = Path(root)
    scans = []

    ply_files = []
    for subdir in ["train", "test", "."]:
        d = base / subdir if subdir != "." else base
        found = sorted(d.glob("*.ply"))
        ply_files.extend(found)
    ply_files = ply_files[::stride]
    print(f"  SensatUrban: {len(ply_files)} files")

    for f in ply_files:
        try:
            v, xyz, rgb, intensity = _load_ply_generic(f)
            print(f"    {f.name}: {len(xyz):,} points", end="")

            label_raw = None
            for key in ("label", "class", "semantic_label", "scalar_Label"):
                if key in v.dtype.names:
                    label_raw = np.asarray(v[key], dtype=np.int64)
                    break
            if label_raw is None:
                print(" [SKIP: no labels]")
                continue

            labels = apply_map(label_raw, SENSATURBAN_MAP)
            scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                    labels=labels))
            print(f" → mapped")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  SensatUrban total: {len(scans)} scans, {total_pts:,} points")
    return scans


def load_parislille(root, stride=1):
    """Load Paris-Lille-3D (mobile LiDAR, French cities).

    Structure: root/*.ply — large PLY files with per-point labels.
    Coarse labels (9 classes): unclassified(0), ground(1), building(2),
    pole(3), bollard(4), trash_can(5), barrier(6), pedestrian(7),
    car(8), natural(9).
    """
    base = Path(root)
    scans = []

    ply_files = sorted(base.glob("*.ply"))
    if not ply_files:
        ply_files = sorted(base.glob("**/*.ply"))
    ply_files = ply_files[::stride]
    print(f"  Paris-Lille-3D: {len(ply_files)} files")

    for f in ply_files:
        try:
            v, xyz, rgb, intensity = _load_ply_generic(f)
            print(f"    {f.name}: {len(xyz):,} points", end="")

            label_raw = None
            for key in ("class", "label", "classification", "scalar_Label"):
                if key in v.dtype.names:
                    label_raw = np.asarray(v[key], dtype=np.int64)
                    break
            if label_raw is None:
                print(" [SKIP: no labels]")
                continue

            labels = apply_map(label_raw, PARISLILLE_MAP)
            scans.append(LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity,
                                    labels=labels))
            print(f" → mapped")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            continue

    total_pts = sum(s.xyz.shape[0] for s in scans) if scans else 0
    print(f"  Paris-Lille-3D total: {len(scans)} scans, {total_pts:,} points")
    return scans


# ============================================================================
# Dataset
# ============================================================================

class PointCloudTileDataset(Dataset):
    """Serves random crops from pre-loaded scans.

    If scan_datasets is provided, tracks which dataset each scan came from
    and returns dataset_id alongside xyz/feats/labels for per-dataset metrics.
    """

    def __init__(self, scans, crop_points=131072, voxel=0.02,
                 augment=True, do_mod_drop=True, steps_per_epoch=1000,
                 scan_datasets=None):
        self.scans = scans
        self.crop = crop_points
        self.voxel = voxel
        self.augment = augment
        self.do_mod_drop = do_mod_drop
        self.steps = steps_per_epoch
        self.scan_datasets = scan_datasets  # list[str], same len as scans

        # Precompute HAG and features for each scan
        print(f"  Precomputing features for {len(scans)} scans...")
        self.prepared = []
        for i, scan in enumerate(scans):
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(scans)}")
            hag = height_above_ground_from_labels(scan.xyz, scan.labels)
            feats = pack_features(scan.rgb, scan.intensity, hag)
            self.prepared.append((scan.xyz.copy(), feats, scan.labels.copy()))
        print(f"  Done precomputing.")

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        # Random scan
        si = np.random.randint(len(self.prepared))
        xyz, feats, labels = self.prepared[si]

        # Voxel downsample
        xyz, feats, labels = self._voxel_ds(xyz, feats, labels)

        # Anchor crop
        xyz, feats, labels = self._anchor_crop(xyz, feats, labels)

        # Center
        xyz = xyz - xyz.mean(axis=0, keepdims=True)

        # Augment
        if self.augment:
            xyz, feats = self._augment(xyz, feats)

        # Modality dropout
        if self.do_mod_drop:
            feats = modality_dropout(feats)

        # Dataset ID for per-dataset tracking
        ds_id = 0
        if self.scan_datasets is not None:
            ds_id = self.scan_datasets[si]

        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(feats).float(),
                torch.from_numpy(labels).long(),
                ds_id)

    def _voxel_ds(self, xyz, feats, labels):
        v = self.voxel
        if v <= 0:
            return xyz, feats, labels
        grid = np.floor(xyz / v).astype(np.int32)
        gmin = grid.min(axis=0)
        grid -= gmin
        dims = grid.max(axis=0).astype(np.int64) + 1
        total = dims[0] * dims[1] * dims[2]
        if total < 2**31:
            k = ((grid[:, 0].astype(np.int32) * np.int32(dims[1])
                  + grid[:, 1]) * np.int32(dims[2]) + grid[:, 2])
        else:
            k = ((grid[:, 0].astype(np.int64) * dims[1]
                  + grid[:, 1]) * dims[2] + grid[:, 2])
        _, pick = np.unique(k, return_index=True)
        return xyz[pick], feats[pick], labels[pick]

    def _anchor_crop(self, xyz, feats, labels):
        N = len(xyz)
        if N == 0:
            # Return zeros
            xyz = np.zeros((self.crop, 3), dtype=np.float32)
            feats = np.zeros((self.crop, FEAT_DIM), dtype=np.float32)
            labels = np.zeros(self.crop, dtype=np.int64)
            return xyz, feats, labels
        if N <= self.crop:
            pad = np.random.randint(0, N, size=self.crop - N)
            idx = np.concatenate([np.arange(N), pad])
        else:
            anchor = xyz[np.random.randint(N)]
            d2 = ((xyz - anchor) ** 2).sum(axis=1)
            idx = np.argpartition(d2, self.crop)[:self.crop]
        return xyz[idx], feats[idx], labels[idx]

    def _augment(self, xyz, feats):
        xyz = xyz.copy()
        # Random yaw rotation
        t = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        # Scale
        xyz *= np.random.uniform(0.9, 1.1)
        # Jitter
        xyz += np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
        # Random flip
        if np.random.rand() < 0.5:
            xyz[:, 0] = -xyz[:, 0]
        return xyz, feats


# ============================================================================
# Collate
# ============================================================================

def collate_fn(batch):
    xyz = torch.stack([b[0] for b in batch])
    feats = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    feats = torch.nan_to_num(feats, nan=0.0)
    ds_ids = [b[3] for b in batch]  # list of dataset IDs (int)
    return xyz, feats, labels, ds_ids


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for xyz, feats, labels, _ds_ids in loader:
        xyz = xyz.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        logits = model(xyz, feats)
        pred = logits.argmax(dim=-1)
        for c in range(1, num_classes):  # skip unlabeled
            p = (pred == c)
            t = (labels == c)
            inter[c] += (p & t).sum().cpu()
            union[c] += (p | t).sum().cpu()

    iou = inter[1:] / union[1:].clamp(min=1)
    per_class = {CLASS_NAMES[i+1]: float(iou[i]) for i in range(len(iou))}
    miou = float(iou[iou > 0].mean()) if (iou > 0).any() else 0.0
    return miou, per_class


# ============================================================================
# Training
# ============================================================================

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    if torch.cuda.is_available():
        print(f"[train] GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[train] VRAM: {vram:.1f} GB")

    out_dir = Path(cfg.get("out_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = cfg.get("num_classes", NUM_CLASSES)
    crop = cfg.get("crop_points", 131072)
    voxel = cfg.get("voxel", 0.02)
    batch_size = cfg.get("batch_size", 16)
    lr = cfg.get("lr", 0.002)
    epochs = cfg.get("epochs", 150)
    steps = cfg.get("steps_per_epoch", 1000)
    num_workers = cfg.get("num_workers", 8)

    # ---- Load datasets (with per-dataset tracking) ----
    print("\n=== Loading datasets ===")
    all_train_scans = []
    all_val_scans = []
    train_ds_ids = []       # parallel list: dataset ID per train scan
    ds_names = []           # ordered list of dataset names
    ds_cfg = cfg.get("datasets", {})

    def _add_train(scans, ds_name):
        """Register train scans with dataset tracking."""
        if ds_name not in ds_names:
            ds_names.append(ds_name)
        ds_id = ds_names.index(ds_name)
        all_train_scans.extend(scans)
        train_ds_ids.extend([ds_id] * len(scans))

    if "toronto3d" in ds_cfg:
        root = ds_cfg["toronto3d"]["root"]
        scans = load_toronto3d(root)
        if scans:
            for s_idx, s in enumerate(scans):
                ply_files = sorted(Path(root).glob("L00*.ply"))
                if s_idx < len(ply_files):
                    name = ply_files[s_idx].stem
                    if name in ("L001", "L003"):
                        _add_train([s], "toronto3d")
                    elif name == "L004":
                        all_val_scans.append(s)
                else:
                    _add_train([s], "toronto3d")

    if "semantickitti" in ds_cfg:
        root = ds_cfg["semantickitti"]["root"]
        stride = cfg.get("kitti_scan_stride", 5)
        train_scans = load_semantickitti(root, train_sequences=[
            "00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            stride=stride)
        val_scans = load_semantickitti(root, train_sequences=["08"],
            stride=stride)
        _add_train(train_scans, "semantickitti")
        all_val_scans.extend(val_scans)

    if "pandaset" in ds_cfg:
        root = ds_cfg["pandaset"]["root"]
        ps_stride = ds_cfg["pandaset"].get("stride", 5)
        ps_scans = load_pandaset(root, stride=ps_stride)
        split = int(len(ps_scans) * 0.8)
        _add_train(ps_scans[:split], "pandaset")
        if split < len(ps_scans):
            all_val_scans.extend(ps_scans[split:])

    if "3dref" in ds_cfg:
        root = ds_cfg["3dref"]["root"]
        ref_stride = ds_cfg["3dref"].get("stride", 1)
        ref_scans = load_3dref(root, stride=ref_stride)
        split = int(len(ref_scans) * 0.8)
        _add_train(ref_scans[:split], "3dref")
        if split < len(ref_scans):
            all_val_scans.extend(ref_scans[split:])

    # ---- NEW DATASETS (residential/European) ----

    if "hessigheim" in ds_cfg:
        root = ds_cfg["hessigheim"]["root"]
        h3d_train = load_hessigheim(root, split="train")
        h3d_val = load_hessigheim(root, split="val")
        _add_train(h3d_train, "hessigheim")
        all_val_scans.extend(h3d_val)

    if "semantic3d" in ds_cfg:
        root = ds_cfg["semantic3d"]["root"]
        s3d_stride = ds_cfg["semantic3d"].get("stride", 1)
        s3d_scans = load_semantic3d(root, stride=s3d_stride)
        split = int(len(s3d_scans) * 0.8)
        _add_train(s3d_scans[:split], "semantic3d")
        if split < len(s3d_scans):
            all_val_scans.extend(s3d_scans[split:])

    if "dales" in ds_cfg:
        root = ds_cfg["dales"]["root"]
        dales_stride = ds_cfg["dales"].get("stride", 1)
        dales_scans = load_dales(root, stride=dales_stride)
        split = int(len(dales_scans) * 0.8)
        _add_train(dales_scans[:split], "dales")
        if split < len(dales_scans):
            all_val_scans.extend(dales_scans[split:])

    if "vaihingen" in ds_cfg:
        root = ds_cfg["vaihingen"]["root"]
        vai_stride = ds_cfg["vaihingen"].get("stride", 1)
        vai_scans = load_vaihingen(root, stride=vai_stride)
        split = int(len(vai_scans) * 0.8)
        _add_train(vai_scans[:split], "vaihingen")
        if split < len(vai_scans):
            all_val_scans.extend(vai_scans[split:])

    if "sensaturban" in ds_cfg:
        root = ds_cfg["sensaturban"]["root"]
        su_stride = ds_cfg["sensaturban"].get("stride", 1)
        su_scans = load_sensaturban(root, stride=su_stride)
        split = int(len(su_scans) * 0.8)
        _add_train(su_scans[:split], "sensaturban")
        if split < len(su_scans):
            all_val_scans.extend(su_scans[split:])

    if "parislille" in ds_cfg:
        root = ds_cfg["parislille"]["root"]
        pl_stride = ds_cfg["parislille"].get("stride", 1)
        pl_scans = load_parislille(root, stride=pl_stride)
        split = int(len(pl_scans) * 0.8)
        _add_train(pl_scans[:split], "parislille")
        if split < len(pl_scans):
            all_val_scans.extend(pl_scans[split:])

    if not all_train_scans:
        raise RuntimeError("No training data! Check dataset paths in config.yaml")

    # Print per-dataset summary
    print(f"\n{'='*50}")
    print(f"{'Dataset':<15} {'Train scans':>12} {'%':>6}")
    print(f"{'-'*50}")
    for di, dname in enumerate(ds_names):
        count = train_ds_ids.count(di)
        pct = 100 * count / len(all_train_scans)
        print(f"  {dname:<13} {count:>10,}  {pct:>5.1f}%")
    print(f"{'-'*50}")
    print(f"  {'TOTAL':<13} {len(all_train_scans):>10,}  100.0%")
    print(f"  Val scans: {len(all_val_scans):,}")
    print(f"{'='*50}\n")

    # ---- Datasets (precompute features, then free raw scans) ----
    train_ds = PointCloudTileDataset(
        all_train_scans, crop_points=crop, voxel=voxel,
        augment=True, do_mod_drop=True, steps_per_epoch=steps,
        scan_datasets=train_ds_ids)
    val_ds = PointCloudTileDataset(
        all_val_scans if all_val_scans else all_train_scans[:1],
        crop_points=crop, voxel=voxel,
        augment=False, do_mod_drop=False, steps_per_epoch=min(100, steps))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    # Free original scan data (already copied into Dataset.prepared)
    del all_train_scans, all_val_scans
    import gc; gc.collect()

    # ---- Model (Point Transformer V3) ----
    ptv3_cfg = cfg.get("ptv3", {})
    model = PointTransformerV3(
        in_feat_dim=cfg.get("in_feat_dim", 5),
        num_classes=num_classes,
        dims=tuple(ptv3_cfg.get("dims", [48, 96, 192, 384])),
        num_heads=tuple(ptv3_cfg.get("num_heads", [3, 6, 12, 24])),
        depths=tuple(ptv3_cfg.get("depths", [2, 2, 6, 2])),
        window_size=ptv3_cfg.get("window_size", 256),
        grid_sizes=tuple(ptv3_cfg.get("grid_sizes", [0.08, 0.16, 0.32])),
        drop=ptv3_cfg.get("drop", 0.0),
        serialize_grid=ptv3_cfg.get("serialize_grid", 0.04),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    lovasz = LovaszSoftmax(ignore_index=0)

    # ---- Train ----
    best_miou = 0.0
    n_datasets = len(ds_names)
    print(f"\n=== Training: {epochs} epochs, {steps} steps/epoch, batch={batch_size} ===")
    print(f"    Tracking loss for {n_datasets} datasets: {', '.join(ds_names)}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Per-dataset loss tracking
        ds_loss_sum = [0.0] * n_datasets
        ds_loss_cnt = [0] * n_datasets
        t0 = time.time()

        for step, (xyz, feats, labels, ds_ids) in enumerate(train_loader):
            xyz = xyz.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(xyz, feats)
                ce = class_weighted_ce(logits, labels)
                lv = lovasz(logits, labels)
                loss = ce + 0.5 * lv

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            total_loss += loss_val

            # Track per-dataset loss (each batch sample may be from different dataset)
            for did in ds_ids:
                if 0 <= did < n_datasets:
                    ds_loss_sum[did] += loss_val
                    ds_loss_cnt[did] += 1

            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={loss_val:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # Evaluate every 5 epochs
        miou = 0.0
        iou_str = ""
        if epoch % 5 == 0 or epoch == 1:
            miou, per_class = evaluate(model, val_loader, num_classes, device)
            iou_str = " | ".join(f"{k}={v:.3f}" for k, v in per_class.items())

        print(f"epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  mIoU={miou:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}  time={elapsed:.0f}s")
        if iou_str:
            print(f"  per-class: {iou_str}")

        # Per-dataset loss report (every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ── per-dataset loss ──")
            ds_losses = []
            for di in range(n_datasets):
                if ds_loss_cnt[di] > 0:
                    avg = ds_loss_sum[di] / ds_loss_cnt[di]
                    ds_losses.append((ds_names[di], avg, ds_loss_cnt[di]))
                else:
                    ds_losses.append((ds_names[di], 0.0, 0))
            # Sort by loss descending (hardest first)
            ds_losses.sort(key=lambda x: -x[1])
            for dname, dloss, dcnt in ds_losses:
                bar = "█" * min(30, int(dloss * 15))
                status = "◀ hardest" if dloss == ds_losses[0][1] and dloss > 0 else ""
                print(f"    {dname:<14} loss={dloss:.4f}  samples={dcnt:>5}  "
                      f"{bar} {status}")

        # Save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "miou": miou,
            "loss": avg_loss,
            "cfg": {
                "num_classes": num_classes,
                "in_feat_dim": cfg.get("in_feat_dim", 5),
                "model": "ptv3",
                "ptv3": ptv3_cfg,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")

        if miou > best_miou:
            best_miou = miou
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  >> New best mIoU: {best_miou:.4f}")

    print(f"\n=== Done. Best mIoU: {best_miou:.4f} ===")
    print(f"Model saved: {out_dir / 'best.pt'}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
