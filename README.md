# Okoliš AI — Trening RandLA-Net

Standalone trening za 8-klasnu semantičku segmentaciju point cloudova.
Koristi Toronto3D + SemanticKITTI datasete.

## Klase (8)

| ID | Klasa       |
|----|-------------|
| 0  | unlabeled   |
| 1  | ground      |
| 2  | road        |
| 3  | sidewalk    |
| 4  | building    |
| 5  | fence       |
| 6  | vegetation  |
| 7  | pole        |

## Datoteke

- `config.yaml` — konfiguracija (batch size, learning rate, putanje dataseta)
- `model.py` — RandLA-Net implementacija
- `losses.py` — Lovász-Softmax + weighted cross-entropy
- `train.py` — cjelokupni trening skript
- `train.ipynb` — Jupyter notebook za RunPod

## Brzi start (RunPod / JupyterLab)

### 1. Pokrenuti RunPod pod

- Template: **RunPod Pytorch 2.x**
- GPU: **RTX A6000 (48 GB VRAM)**
- Disk: **100+ GB** (za datasete)

### 2. Otvoriti terminal ili train.ipynb u JupyterLabu

### 3. Instalirati zavisnosti

```bash
pip install plyfile pyyaml scipy
```

### 4. Klonirati repo

```bash
cd /workspace
git clone https://github.com/TVOJ_USERNAME/okolis-training.git
cd okolis-training
```

### 5. Pripremiti datasete

**Toronto3D** (slobodan pristup):
```bash
mkdir -p /workspace/data/Toronto_3D
cd /workspace/data/Toronto_3D
# Preuzmi sa Zenodo:
wget https://zenodo.org/records/4570104/files/Toronto_3D.zip
unzip Toronto_3D.zip
```

**SemanticKITTI**:
1. Registriraj se na https://www.semantic-kitti.org/
2. Prihvati uvjete korištenja
3. Preuzmi KITTI Odometry Velodyne data sa http://www.cvlibs.net/datasets/kitti/eval_odometry.php
4. Preuzmi SemanticKITTI label data sa semantic-kitti.org
5. Raspakiraj u `/workspace/data/SemanticKITTI/dataset/sequences/`

Očekivana struktura:
```
/workspace/data/SemanticKITTI/dataset/sequences/
├── 00/
│   ├── velodyne/
│   │   ├── 000000.bin
│   │   └── ...
│   └── labels/
│       ├── 000000.label
│       └── ...
├── 01/
└── ...
```

### 6. Pokrenuti trening

```bash
cd /workspace/okolis-training
python train.py --config config.yaml
```

## Konfiguracija (RTX A6000 48GB)

Ključni parametri u `config.yaml`:

- `batch_size: 16` — optimalno za 48 GB VRAM
- `crop_points: 131072` — 128K točaka po uzorku
- `voxel: 0.02` — voxel veličina za downsample
- `kitti_scan_stride: 5` — učitava svaki 5. KITTI sken (štedi RAM)
- `epochs: 150` — ukupno epoha
- `lr: 0.002` — AdamW learning rate sa cosine decay

## Izlaz

Modeli se spremaju u `/workspace/runs/rtx_a6000/`:
- `last.pt` — zadnji checkpoint
- `best.pt` — checkpoint sa najboljim mIoU

Checkpoint sadrži:
```python
{
    "epoch": int,
    "model": state_dict,
    "optimizer": state_dict,
    "miou": float,
    "loss": float,
    "cfg": {"num_classes": 8, "in_feat_dim": 5}
}
```

## Korištenje modela

Nakon treninga, preuzmi `best.pt` i koristi ga u glavnom Okoliš AI projektu s `test_scan.py`:
```bash
python test_scan.py best.pt tvoj_sken.ply --export
```
