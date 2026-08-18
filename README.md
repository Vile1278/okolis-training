# Okoliš AI — PTv3 trening

Standalone trening za 8-klasnu semantičku segmentaciju point cloudova.
**Aktivnih 5 dataseta:** Hessigheim3D, Semantic3D, Paris-Lille-3D, Toronto3D, SemanticKITTI.
(Pandaset i SensatUrban su isključeni — NE skidati, štedi ~50 GB.)

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

- `config.yaml` — konfiguracija (dataseti, weightovi, hiperparametri)
- `model.py` — Point Transformer V3 (s decoder fixom, kolovoz 2026.)
- `losses.py` — Lovász-Softmax + weighted cross-entropy
- `train.py` — trening skript (lazy cache, weighted sampling, manifest)
- `analyze_classes.py` — analiza distribucije klasa u cacheu

## Brzi start (RunPod)

### 1. Kreirati pod

- Template: **RunPod PyTorch 2.x**
- GPU: **RTX A6000 (48 GB VRAM)**
- **VAŽNO: dodati Network Volume (150+ GB) montiran na /workspace** —
  bez toga terminiranje poda briše sve datasete i checkpointe!

### 2. Setup

```bash
cd /workspace
git clone https://github.com/Vile1278/okolis-training.git
cd okolis-training
pip install plyfile pyyaml scipy pandas gdown "laspy[lazrs]"
mkdir -p /workspace/data
```

### 3. Dataseti (5 potrebnih, ~120 GB, ~2-3 h)

**SemanticKITTI (~80 GB — pokrenuti PRVI, u pozadini):**
```bash
mkdir -p /workspace/data/SemanticKITTI && cd /workspace/data/SemanticKITTI
nohup wget -c "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" -O velodyne.zip > kitti_dl.log 2>&1 &
wget "https://www.semantic-kitti.org/assets/data_odometry_labels.zip" -O labels.zip
# kad velodyne završi:
python -c "import zipfile; zipfile.ZipFile('velodyne.zip').extractall('.')"
python -c "import zipfile; zipfile.ZipFile('labels.zip').extractall('.')"
rm velodyne.zip labels.zip
```

**Semantic3D (~30 GB):**
```bash
mkdir -p /workspace/data/Semantic3D && cd /workspace/data/Semantic3D
for f in bildstein_station1_xyz_intensity_rgb bildstein_station3_xyz_intensity_rgb \
         bildstein_station5_xyz_intensity_rgb domfountain_station1_xyz_intensity_rgb \
         domfountain_station2_xyz_intensity_rgb domfountain_station3_xyz_intensity_rgb \
         neugasse_station1_xyz_intensity_rgb sg27_station1_intensity_rgb \
         sg27_station2_intensity_rgb sg27_station4_intensity_rgb \
         sg27_station5_intensity_rgb sg27_station9_intensity_rgb \
         sg28_station4_intensity_rgb untermaederbrunnen_station1_xyz_intensity_rgb \
         untermaederbrunnen_station3_xyz_intensity_rgb; do
  wget -c "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/$f.7z"
done
wget -c https://share.phys.ethz.ch/~pf/semantic3d/data/sem8_labels_training.7z
apt-get update && apt-get install -y p7zip-full
for f in *.7z; do 7z x "$f" && rm "$f"; done
```

**Toronto3D (~4 GB, Kaggle):**
```bash
# jednom: postaviti kaggle.json (kaggle.com → Account → Create API Token)
mkdir -p /root/.kaggle
echo '{"username":"TVOJ_USERNAME","key":"TVOJ_KEY"}' > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
pip install kaggle

mkdir -p /workspace/data/Toronto_3D && cd /workspace/data/Toronto_3D
kaggle datasets download -d priteshraj10/point-cloud-lidar-toronto-3d
python -c "import zipfile; zipfile.ZipFile('point-cloud-lidar-toronto-3d.zip').extractall('.')"
rm point-cloud-lidar-toronto-3d.zip
```

**Paris-Lille-3D (~2 GB):** ručno kroz browser (ownCloud Mines ParisTech,
lozinka `Paris-Lille-3D`), pa upload ZIP-a kroz RunPod file browser:
```bash
apt-get install -y unzip
mkdir -p /workspace/data/ParisLille3D && cd /workspace/data/ParisLille3D
unzip ParisLille3D.zip && rm ParisLille3D.zip
# treba završiti s: Lille1_1.ply, Lille1_2.ply, Lille2.ply, Paris.ply
```

**Hessigheim 3D (~2 GB):** upload s lokalnog računala
(`dataset/Hessigheim_Benchmark/Epoch_March2019/LiDAR/` — samo .laz fajlovi):
```bash
mkdir -p /workspace/data/Hessigheim3D
# uploadaj kroz RunPod file browser: Mar19_train.laz, Mar19_val.laz
# u /workspace/data/Hessigheim3D/
```

### 4. Pokrenuti trening

```bash
cd /workspace/okolis-training
nohup python train.py --config config.yaml > train.log 2>&1 &
tail -f train.log        # Ctrl+C prekida samo gledanje, ne trening
```

- Prvi start: gradi disk cache (~45-60 min) i sprema `manifest.json`
- Svaki sljedeći start: kreće iz cachea za par sekundi (sirovi podaci
  više nisu potrebni — smiju se i obrisati nakon prvog starta)
- Epoha traje ~50 min; evaluacija (mIoU + per-class) svaku 5. epohu
- Stop: kad mIoU stagnira 10 epoha (`kill %1` ili Ctrl+C ako je u forgroundu)

## Konfiguracija (config.yaml, trenutno stanje)

- `crop_points: 65536`, `batch_size: 2` — 65K točaka po cropu (A6000 48 GB)
- `lr: 0.0003` + warmup 3 epohe + cosine decay
- `kitti_scan_stride: 20` — svaki 20. KITTI scan
- weightovi dataseta: hessigheim 2.0, semantic3d 1.5, parislille 1.2, toronto3d 1.0, semantickitti 0.4
- class weights u lossu: rijetke klase (sidewalk/fence/vehicle) pojačane
- rare-class anchoring: 35% cropova centrirano na rijetku klasu
- `out_dir: /workspace/runs/ptv3_v2_fixed`

## Izlaz

- `best.pt` — najbolji mIoU (koristiti za inference/fine-tuning)
- `last.pt` — zadnja epoha (za resume: `resume_from` u config.yaml)
- **best.pt skinuti lokalno nakon treninga!**

## Korištenje modela

U okolis-test folderu glavnog projekta:
```bash
python test_pipeline.py --weights best.pt --ply tvoj_sken.ply --out rezultat.ply
```
