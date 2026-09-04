#!/usr/bin/env python3
"""E2: fino podešavanje PTv3 na iPhone snimci kod_Tina s prostornom unakrsnom
validacijom (4 bloka = 4 folda). Ne mijenja train.py — za svaki fold napravi
mapu s .npz datotekama, generira config i pozove `python train.py --config`.

Za svaki fold k (blok k = testni):
  1. trening blokovi (3) se spreme kao 6 .npz datoteka (svaki blok u 2 polovice)
     + testni blok kao 'z_test_block.npz' → train.py ga uzima kao val (samo za
     praćenje krivulje; NE koristi se za odabir checkpointa)
  2. train.py fino podešava od --base checkpointa (best_final.pt), fiksan broj epoha
  3. zadnji checkpoint (last.pt, EMA težine) i početni (zero-shot) se evaluiraju
     na testnom bloku ISTIM postupkom (preklapajući cropovi od crop_points točaka)
  4. sprema fold{k}_result.json (IoU po klasi, mIoU, matrica zabune) i
     fold{k}_probs.npz (xyz + softmax vjerojatnosti testnog bloka, za fusion)

Pokretanje na podu (iz mape okolis-training):
  python finetune_folds.py --npz /workspace/data/kod_Tina_train.npz \
      --base /workspace/runs/ptv3_big2/best.pt --work /workspace/e2 --folds 0 1 2 3
Samo evaluacija (bez treninga), npr. zero-shot:  --no_train
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CLASS_NAMES = ["unlabeled", "ground", "(road→ground)", "sidewalk",
               "building", "wall", "vegetation", "(vehicle izbačen)"]
EVAL = {1: "ground", 3: "sidewalk", 4: "building", 5: "wall", 6: "vegetation"}
NEW_MAP = {0: 0, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0}


def remap(lab):
    out = np.zeros_like(lab)
    for k, v in NEW_MAP.items():
        out[lab == k] = v
    return out


# ---------------------------------------------------------------- HAG bez oznaka
def hag_label_free(xyz, cell=1.0):
    """Visina iznad tla iz minimuma z po ćeliji (bez oznaka) — kao u inferenceu."""
    from scipy.ndimage import median_filter
    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    nx = max(1, int(np.ceil((xyz[:, 0].max() - xmin) / cell)))
    ny = max(1, int(np.ceil((xyz[:, 1].max() - ymin) / cell)))
    ci = np.clip(((xyz[:, 0] - xmin) / cell).astype(int), 0, nx - 1)
    cj = np.clip(((xyz[:, 1] - ymin) / cell).astype(int), 0, ny - 1)
    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    np.minimum.at(grid, (ci, cj), xyz[:, 2].astype(np.float32))
    grid[np.isinf(grid)] = np.nan
    for _ in range(5):
        if not np.any(np.isnan(grid)):
            break
        med = np.nanmedian(grid)
        filled = median_filter(np.nan_to_num(grid, nan=med), size=3)
        m = np.isnan(grid); grid[m] = filled[m]
    hag = xyz[:, 2] - grid[ci, cj]
    return np.clip(hag, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------- blokovi
def spatial_blocks(xyz):
    """2×2 podjela po medijanu x i y (približno jednak broj točaka po bloku)."""
    mx, my = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    bx = (xyz[:, 0] >= mx).astype(int)
    by = (xyz[:, 1] >= my).astype(int)
    return bx * 2 + by, (float(mx), float(my))


def write_fold_data(xyz, rgb, labels, block, k, out_dir):
    """Zapiše train .npz-ove (3 bloka × 2 polovice) + testni blok kao zadnji (val)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.npz"):
        f.unlink()
    n_train = 0
    for b in range(4):
        m = block == b
        if b == k:
            np.savez_compressed(out_dir / "z_test_block.npz", xyz=xyz[m], rgb=rgb[m], labels=labels[m])
            continue
        # polovice po dominantnoj osi bloka (da datoteke budu slične veličine)
        sub = xyz[m]
        axis = 0 if np.ptp(sub[:, 0]) >= np.ptp(sub[:, 1]) else 1
        med = np.median(sub[:, axis])
        for h, mask in enumerate([sub[:, axis] < med, sub[:, axis] >= med]):
            idx = np.flatnonzero(m)[mask]
            np.savez_compressed(out_dir / f"train_b{b}_h{h}.npz", xyz=xyz[idx], rgb=rgb[idx], labels=labels[idx])
            n_train += len(idx)
    return n_train, int((block == k).sum())


# ---------------------------------------------------------------- model / inference
def load_model(ckpt_path, device, key="model"):
    """key="model" = EMA težine (produkcijske), key="model_raw" = sirove težine."""
    import torch
    from model import PointTransformerV3
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {}); p = cfg.get("ptv3", {})
    model = PointTransformerV3(
        in_feat_dim=cfg.get("in_feat_dim", 5), num_classes=cfg.get("num_classes", 8),
        dims=tuple(p.get("dims", [64, 128, 256, 512])), num_heads=tuple(p.get("num_heads", [4, 8, 16, 32])),
        depths=tuple(p.get("depths", [2, 2, 8, 2])), window_size=p.get("window_size", 256),
        grid_sizes=tuple(p.get("grid_sizes", [0.08, 0.16, 0.32])), drop=0.0,
        serialize_grid=p.get("serialize_grid", 0.04), multi_curve=p.get("multi_curve", True)).to(device)
    state = ck.get(key, ck.get("model"))
    model.load_state_dict(state)
    model.eval()
    return model, ck.get("epoch", "?"), float(ck.get("miou", 0.0))


def predict_block(model, xyz, feats, crop, device, seed=0, extra_passes=1):
    """Preklapajući cropovi od `crop` točaka (kao _anchor_crop u treningu):
    1. pokrivanje: dok ima nepokrivenih točaka, sidro = nepokrivena točka;
    2. dodatni prolazi sa sidrima na mreži (glađe granice).
    Vraća prosjek softmax vjerojatnosti (N, C)."""
    import torch
    from scipy.spatial import cKDTree
    N = len(xyz); C = model.num_classes
    probs = np.zeros((N, C), dtype=np.float32); cnt = np.zeros(N, dtype=np.float32)
    tree = cKDTree(xyz)
    rng = np.random.default_rng(seed)

    def run(idx):
        sub = xyz[idx] - xyz[idx].mean(axis=0, keepdims=True)
        with torch.no_grad():
            x = torch.from_numpy(sub).float().unsqueeze(0).to(device)
            f = torch.from_numpy(feats[idx]).float().unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x, f)
            p = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        probs[idx] += p; cnt[idx] += 1

    def crop_idx(anchor_xyz):
        if N <= crop:
            pad = rng.integers(0, N, size=crop - N)
            return np.concatenate([np.arange(N), pad])
        _, idx = tree.query(anchor_xyz, k=crop)
        return np.asarray(idx)

    # 1. pokrivanje
    covered = np.zeros(N, dtype=bool); n_crops = 0
    while not covered.all():
        a = rng.choice(np.flatnonzero(~covered))
        idx = crop_idx(xyz[a]); run(idx); covered[idx] = True; n_crops += 1
    # 2. dodatni prolazi: sidra na pravilnoj mreži
    for _ in range(extra_passes):
        step = max(1.0, np.sqrt(crop / max(N, 1) * (np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 1]))) * 0.5)
        xs = np.arange(xyz[:, 0].min(), xyz[:, 0].max() + step, step)
        ys = np.arange(xyz[:, 1].min(), xyz[:, 1].max() + step, step)
        for gx in xs:
            for gy in ys:
                _, near = tree.query([gx, gy, np.median(xyz[:, 2])], k=1)
                run(crop_idx(xyz[near])); n_crops += 1
    probs /= np.maximum(cnt, 1)[:, None]
    return probs, n_crops


def metrics(gt, pred, C=8):
    m = gt != 0
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (gt[m], pred[m]), 1)
    res = {"per_class": {}, "confusion": cm.tolist()}
    ious = []
    for c, name in EVAL.items():
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        u = tp + fp + fn
        iou = float(tp / u) if u > 0 else None
        res["per_class"][name] = {"iou": None if iou is None else round(iou, 4), "gt_points": int(cm[c, :].sum()),
                                  "pred_points": int(cm[:, c].sum())}
        if cm[c, :].sum() > 0: ious.append(iou)
    res["mIoU_present"] = round(float(np.mean(ious)), 4)
    res["OA"] = round(float(np.trace(cm) / max(cm.sum(), 1)), 4)
    res["n_eval_points"] = int(cm.sum())
    return res


# ---------------------------------------------------------------- glavni tok
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="kod_Tina_train.npz (xyz, rgb, labels)")
    ap.add_argument("--base", required=True, help="početni checkpoint (best_final.pt)")
    ap.add_argument("--work", default="e2_work")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200, help="steps_per_epoch (batch 2 → steps/2 iteracija)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=49152)
    ap.add_argument("--template", default=str(HERE / "config.yaml"))
    ap.add_argument("--no_train", action="store_true", help="samo zero-shot evaluacija")
    ap.add_argument("--cpu_test", action="store_true", help="brzi test logike na CPU-u (mali crop, 1 blok)")
    a = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)

    d = np.load(a.npz)
    xyz = d["xyz"].astype(np.float32); rgb = d["rgb"].astype(np.float32); labels = remap(d["labels"].astype(np.int64))
    xyz = xyz - xyz.mean(axis=0, keepdims=True)  # globalno centriranje
    block, (mx, my) = spatial_blocks(xyz)
    info = {"n_points": int(len(xyz)), "split": {"median_x": mx, "median_y": my},
            "blocks": {int(b): {"n": int((block == b).sum()),
                                "classes": {EVAL[c]: int(((block == b) & (labels == c)).sum()) for c in EVAL}}
                       for b in range(4)}}
    print(json.dumps(info, indent=1))
    json.dump(info, open(work / "blocks.json", "w"), indent=1)

    hag = hag_label_free(xyz)
    feats = np.zeros((len(xyz), 5), dtype=np.float32)
    feats[:, 0:3] = rgb[:, :3]; feats[:, 4] = hag   # intenzitet = 0 (iPhone ga nema)

    tmpl = yaml.safe_load(open(a.template))
    crop = 4096 if a.cpu_test else a.crop

    base_model, base_ep, base_miou = load_model(a.base, device)
    print(f"[base] {a.base}: epoch {base_ep}, val mIoU {base_miou:.4f}")
    # train.py za resume uzima "model_raw" (sirove težine) — best_final.pt ima u
    # "model_raw" ožiljak agresivne epohe (vegetation → 0), a rekord 0,4163 su EMA
    # težine u "model". Zato treningu dajemo EMA težine i svjež optimizator.
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    base_ema_path = work / "base_ema.pt"
    torch.save({"model": ck["model"], "model_raw": ck["model"], "epoch": ck.get("epoch", 0),
                "miou": ck.get("miou", 0.0), "cfg": ck.get("cfg", {})}, base_ema_path)
    print(f"[base] EMA težine spremljene kao početna točka treninga: {base_ema_path}")

    for k in a.folds:
        t0 = time.time()
        fold_dir = work / f"fold{k}"; fold_dir.mkdir(exist_ok=True)
        data_dir = fold_dir / "data"
        n_tr, n_te = write_fold_data(xyz, rgb, labels, block, k, data_dir)
        print(f"\n=== FOLD {k}: train {n_tr:,} točaka, test {n_te:,} točaka ===")
        test_m = block == k
        txyz, tfeats, tlab = xyz[test_m], feats[test_m], labels[test_m]
        if a.cpu_test:
            pick = np.random.default_rng(0).choice(len(txyz), min(20000, len(txyz)), replace=False)
            txyz, tfeats, tlab = txyz[pick], tfeats[pick], tlab[pick]

        # --- zero-shot na testnom bloku
        probs0, nc0 = predict_block(base_model, txyz, tfeats, crop, device)
        r0 = metrics(tlab, probs0.argmax(1)); r0["n_crops"] = nc0
        print(f"  zero-shot: mIoU {r0['mIoU_present']}  " + "  ".join(f"{n}={v['iou']}" for n, v in r0["per_class"].items()))
        result = {"fold": k, "test_block": k, "n_train": n_tr, "n_test": n_te, "zero_shot": r0}
        np.savez_compressed(fold_dir / f"fold{k}_probs_zeroshot.npz", xyz=txyz, probs=probs0.astype(np.float16), labels=tlab)

        # --- fino podešavanje (train.py, bez izmjena)
        if not a.no_train:
            cfg = dict(tmpl)
            cfg["datasets"] = {"iphone": {"root": str(data_dir.resolve()), "weight": 1.0}}
            cfg["out_dir"] = str((fold_dir / "run").resolve())
            cfg["cache_dir"] = str((fold_dir / "cache").resolve())
            cfg["resume_from"] = str(base_ema_path.resolve())
            cfg["ema_decay"] = 0.99   # kratki run: EMA se mora stići približiti novim težinama
            cfg["epochs"] = a.epochs; cfg["steps_per_epoch"] = a.steps; cfg["lr"] = a.lr
            cfg["crop_points"] = crop; cfg["batch_size"] = 2; cfg["grad_accum"] = 1
            cfg["num_workers"] = 0
            cfg_path = fold_dir / f"config_fold{k}.yaml"
            yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False, allow_unicode=True)
            log = fold_dir / f"train_fold{k}.log"
            print(f"  train.py → {log}")
            with open(log, "w") as lf:
                rc = subprocess.call([sys.executable, "-u", str(HERE / "train.py"), "--config", str(cfg_path)],
                                     stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
            if rc != 0:
                print(f"  [ERROR] train.py rc={rc} — vidi {log}"); result["train_rc"] = rc
            last = fold_dir / "run" / "last.pt"
            if last.exists():
                for key, tag in [("model_raw", "raw"), ("model", "ema")]:
                    ft_model, ft_ep, ft_val = load_model(last, device, key=key)
                    probs1, nc1 = predict_block(ft_model, txyz, tfeats, crop, device)
                    r1 = metrics(tlab, probs1.argmax(1)); r1["n_crops"] = nc1
                    r1["checkpoint"] = f"last.pt[{key}]"; r1["epoch"] = ft_ep; r1["val_miou_trainpy"] = ft_val
                    print(f"  fine-tuned ({tag}): mIoU {r1['mIoU_present']}  " + "  ".join(f"{n}={v['iou']}" for n, v in r1["per_class"].items()))
                    result[f"fine_tuned_{tag}"] = r1
                    np.savez_compressed(fold_dir / f"fold{k}_probs_finetuned_{tag}.npz", xyz=txyz, probs=probs1.astype(np.float16), labels=tlab)
                    del ft_model
                # krivulja iz loga
                curve = []
                for line in open(log, encoding="utf-8", errors="ignore"):
                    if line.startswith("epoch") and "mIoU=" in line:
                        curve.append(line.strip())
                result["train_curve"] = curve
        result["seconds"] = round(time.time() - t0, 1)
        json.dump(result, open(fold_dir / f"fold{k}_result.json", "w"), indent=1, ensure_ascii=False)
        print(f"  spremljeno {fold_dir / f'fold{k}_result.json'} ({result['seconds']} s)")

    # sažetak
    summary = {}
    for k in a.folds:
        p = work / f"fold{k}" / f"fold{k}_result.json"
        if p.exists(): summary[f"fold{k}"] = json.load(open(p))
    json.dump(summary, open(work / "e2_summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nSAŽETAK: {work / 'e2_summary.json'}")


if __name__ == "__main__":
    main()
