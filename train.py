"""Standalone training script for Okolis AI.

Trains RandLA-Net on Toronto3D + SemanticKITTI for 8-class segmentation.
All helpers are inline -- no imports from any parent project.

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
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from model import RandLANet
from losses import LovaszSoftmax, class_weighted_ce


# ============================================================================
# Label Maps (8 classes)
# ============================================================================
NUM_CLASSES = 8
CLASS_NAMES = [
    "unlabeled", "ground", "road", "sidewalk",
    "building", "fence", "vegetation", "pole",
]

TORONTO3D_MAP = {
    0: 0, 1: 2, 2: 2, 3: 6, 4: 4, 5: 7, 6: 7, 7: 0, 8: 5,
}

# SemanticKITTI RAW label -> unified 8-class
# Note: KITTI uses raw label IDs (not learning IDs), so we map those directly
SEMKITTI_RAW_MAP = {
    0: 0, 1: 0,
    10: 0, 11: 0, 13: 0, 15: 0, 16: 0, 18: 0, 20: 0,  # vehicles -> unlabeled
    30: 0, 31: 0, 32: 0,                                  # persons -> unlabeled
    40: 2, 44: 3, 48: 3, 49: 1,                            # road/parking/sidewalk/other-ground
    50: 4, 51: 5, 52: 0,                                   # building/fence/other-struct
    60: 2,                                                  # lane-marking -> road
    70: 6, 71: 6, 72: 1,                                   # vegetation/trunk/terrain
    80: 7, 81: 7,                                           # pole/traffic-sign
    99: 0,
    252: 0, 253: 0, 254: 0, 255: 0, 256: 0, 257: 0, 258: 0, 259: 0,
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


# ============================================================================
# Dataset
# ============================================================================

class PointCloudTileDataset(Dataset):
    """Serves random crops from pre-loaded scans."""

    def __init__(self, scans, crop_points=131072, voxel=0.02,
                 augment=True, do_mod_drop=True, steps_per_epoch=1000):
        self.scans = scans
        self.crop = crop_points
        self.voxel = voxel
        self.augment = augment
        self.do_mod_drop = do_mod_drop
        self.steps = steps_per_epoch

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

        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(feats).float(),
                torch.from_numpy(labels).long())

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
    return xyz, feats, labels


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for xyz, feats, labels in loader:
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

    # ---- Load datasets ----
    print("\n=== Loading datasets ===")
    all_train_scans = []
    all_val_scans = []
    ds_cfg = cfg.get("datasets", {})

    if "toronto3d" in ds_cfg:
        root = ds_cfg["toronto3d"]["root"]
        scans = load_toronto3d(root)
        if scans:
            # L001, L003 = train; L004 = val; L002 = test (skip)
            for s_idx, s in enumerate(scans):
                ply_files = sorted(Path(root).glob("L00*.ply"))
                if s_idx < len(ply_files):
                    name = ply_files[s_idx].stem
                    if name in ("L001", "L003"):
                        all_train_scans.append(s)
                    elif name == "L004":
                        all_val_scans.append(s)
                    # L002 = test, skip
                else:
                    all_train_scans.append(s)

    if "semantickitti" in ds_cfg:
        root = ds_cfg["semantickitti"]["root"]
        stride = cfg.get("kitti_scan_stride", 5)
        train_scans = load_semantickitti(root, train_sequences=[
            "00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            stride=stride)
        val_scans = load_semantickitti(root, train_sequences=["08"],
            stride=stride)
        all_train_scans.extend(train_scans)
        all_val_scans.extend(val_scans)

    if not all_train_scans:
        raise RuntimeError("No training data! Check dataset paths in config.yaml")

    print(f"\nTotal: {len(all_train_scans)} train scans, {len(all_val_scans)} val scans")

    # ---- Datasets (precompute features, then free raw scans) ----
    train_ds = PointCloudTileDataset(
        all_train_scans, crop_points=crop, voxel=voxel,
        augment=True, do_mod_drop=True, steps_per_epoch=steps)
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

    # ---- Model ----
    model = RandLANet(
        in_feat_dim=cfg.get("in_feat_dim", 5),
        num_classes=num_classes,
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
    print(f"\n=== Training: {epochs} epochs, {steps} steps/epoch, batch={batch_size} ===\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (xyz, feats, labels) in enumerate(train_loader):
            xyz = xyz.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(xyz, feats)
                ce = class_weighted_ce(logits, labels)
                lv = lovasz(logits, labels)
                loss = ce + 0.5 * lv

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

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

        # Save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "miou": miou,
            "loss": avg_loss,
            "cfg": {"num_classes": num_classes, "in_feat_dim": cfg.get("in_feat_dim", 5)},
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
