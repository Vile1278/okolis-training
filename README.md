# Okoliš AI — PTv3

Standalone trening za 8-klasnu semantičku segmentaciju point cloudova.
Koristi Toronto3D + SemanticKITTI + Pandaset + Hessigheim3D + Semantic3D + SensatUrban + Paris-Lille-3D datasete.

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
| 7  | vehicle     |

## Datoteke

- `config.yaml` — konfiguracija (batch size, learning rate, putanje dataseta)
- `model.py` — Point Transformer V3 implementacija
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
git clone https://github.com/Vile1278/okolis-training.git
cd okolis-training
```

### 5. Pripremiti datasete

**Kaggle** 
```bash
pip install kaggle gdown laspy[lazrs]
```

**Kaggle** 
```bash
mkdir /root/.kaggle
echo '{"username":"TVOJ_USERNAME","key":"TVOJ_KEY"}' > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
```

**data dir** 
```bash
mkdir /workspace/data
```

**Semantic3D**
```bash

mkdir -p /workspace/data/Semantic3D && cd /workspace/data/Semantic3D

# Training point clouds (~12 GB komprimirano)
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station1_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station3_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station5_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station1_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station2_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station3_xyz_intensity_rgb.7z &
wait
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/neugasse_station1_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station1_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station2_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station4_intensity_rgb.7z &
wait
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station5_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station9_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg28_station4_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/untermaederbrunnen_station1_xyz_intensity_rgb.7z &
wget https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/untermaederbrunnen_station3_xyz_intensity_rgb.7z &
wait

# Labeli
wget https://share.phys.ethz.ch/~pf/semantic3d/data/sem8_labels_training.7z

# Raspakiraj sve i očisti
apt-get update && apt-get install -y p7zip-full
for f in *.7z; do 7z x "$f" && rm "$f"; done

echo "DONE — $(ls *.txt | wc -l) txt files, $(du -sh . | cut -f1) total"

cd /workspace/data/Semantic3D && rm -f wget-log*
ls -la *.txt | wc -l
ls -la *.labels | wc -l
du -sh .
```

**Paris-Lille-3D**
```bash
apt-get install -y unzip
mkdir -p /workspace/data/ParisLille3D && cd /workspace/data/ParisLille3D
gdown "TVOJ_GOOGLE_DRIVE_LINK" -O ParisLille3D.zip
unzip ParisLille3D.zip && rm ParisLille3D.zip
```

**Hessigheim 3D**
```bash
mkdir -p /workspace/data/Hessigheim3D
cd /workspace/data/Hessigheim3D
gdown "LINK_TRAIN" -O Mar19_train.laz
gdown "LINK_VAL" -O Mar19_val.laz
gdown "LINK_TEST" -O Mar19_test_GroundTruth.laz
```

**SensatUrban**
```bash
mkdir -p /workspace/data/SensatUrban && cd /workspace/data/SensatUrban
gdown --folder "https://drive.google.com/drive/folders/1xd6oc0yJFQ74r54zVJCTGypohvv7ajXG"
unzip sensaturban.zip
```


**Pandaset**
```bash
cd /workspace/data
kaggle datasets download -d usharengaraju/pandaset-dataset --unzip -p Pandaset
```

**Toronto3D** (slobodan pristup):
```bash
mkdir /workspace/data/Toronto_3D
cd /workspace/data/Toronto_3D
kaggle datasets download -d priteshraj10/point-cloud-lidar-toronto-3d
python -c "import zipfile; zipfile.ZipFile('point-cloud-lidar-toronto-3d.zip').extractall('.')"
ls *.ply
```

**SemanticKITTI**:
```bash
mkdir -p /workspace/data/SemanticKITTI
cd /workspace/data/SemanticKITTI
wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" -O velodyne.zip
wget "https://www.semantic-kitti.org/assets/data_odometry_labels.zip" -O labels.zip
python -c "import zipfile; zipfile.ZipFile('velodyne.zip').extractall('.')"
python -c "import zipfile; zipfile.ZipFile('labels.zip').extractall('.')"
```


provjera strukture:
```bash
ls dataset/sequences/00/velodyne/ | head -3
ls dataset/sequences/00/labels/ | head -3
```


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

- `batch_size: 4` — optimalno za 48 GB VRAM
- `crop_points: 65536` — 65K točaka po uzorku
- `voxel: 0.02` — voxel veličina za downsample
- `kitti_scan_stride: 5` — učitava svaki 5. KITTI sken (štedi RAM)
- `epochs: 150` — ukupno epoha
- `lr: 0.0004` — AdamW learning rate sa cosine decay

## Izlaz

Modeli se spremaju u `/workspace/runs/ptv3_a6000/`:
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
