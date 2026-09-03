# Okoliš AI — PTv3 trening

Trening semantičke segmentacije point cloudova (Point Transformer V3, ~28M parametara).
**Aktualni setup (Run 6):** 3 europska dataseta, nastavak od checkpointa `best_final.pt`.

## Klase (aktualna taksonomija)

| ID | Klasa       | Napomena |
|----|-------------|----------|
| 0  | unlabeled   | |
| 1  | ground      | road spojen u ground |
| 2  | —           | ne koristi se |
| 3  | sidewalk    | |
| 4  | building    | |
| 5  | wall        | zidići/suhozidi (bivši fence) |
| 6  | vegetation  | |
| 7  | —           | vehicle izbačen |

## Postavljanje poda — KORAK PO KORAK

### 0. RunPod pod
- Template: **RunPod Pytorch 2.x**, GPU: **RTX A6000 (48 GB)**, disk: **100+ GB**
- **PREPORUKA: Network Volume montiran na /workspace** — bez njega svako
  gašenje poda briše datasete, cache i checkpointe (već nas dvaput koštalo).

### 1. Paketi + repo
```bash
pip install plyfile pyyaml scipy laspy[lazrs] gdown
apt-get update && apt-get install -y p7zip-full
cd /workspace
git clone https://github.com/Vile1278/okolis-training.git
mkdir -p /workspace/data
```

### 2. UPLOAD CHECKPOINTA (novo — bez ovoga nema nastavka treninga!)
S laptopa uploadaj `best_final.pt` (228 MB, iz okolis-test foldera) na pod,
pa ga stavi na putanju koju config očekuje:
```bash
mkdir -p /workspace/runs/ptv3_big2
mv /workspace/best_final.pt /workspace/runs/ptv3_big2/best.pt
ls -lh /workspace/runs/ptv3_big2/best.pt   # mora biti ~228 MB
```
(Upload: JupyterLab file browser → drag&drop u /workspace, traje par minuta.)

### 3. Hessigheim3D (~5 GB) — NAJVAŽNIJI (jedini izvor wall i sidewalk)
```bash
mkdir -p /workspace/data/Hessigheim3D && cd /workspace/data/Hessigheim3D
gdown "LINK_TRAIN" -O Mar19_train.laz
gdown "LINK_VAL"   -O Mar19_val.laz
gdown "LINK_TEST"  -O Mar19_test_GroundTruth.laz   # OBAVEZNO — od Runa 5 ide u trening!
```

### 4. Semantic3D (~12 GB komprimirano)
```bash
mkdir -p /workspace/data/Semantic3D && cd /workspace/data/Semantic3D
B=https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1
wget $B/bildstein_station1_xyz_intensity_rgb.7z $B/bildstein_station3_xyz_intensity_rgb.7z &
wget $B/bildstein_station5_xyz_intensity_rgb.7z $B/domfountain_station1_xyz_intensity_rgb.7z &
wget $B/domfountain_station2_xyz_intensity_rgb.7z $B/domfountain_station3_xyz_intensity_rgb.7z &
wget $B/neugasse_station1_xyz_intensity_rgb.7z $B/sg27_station1_intensity_rgb.7z &
wait
wget $B/sg27_station2_intensity_rgb.7z $B/sg27_station4_intensity_rgb.7z &
wget $B/sg27_station5_intensity_rgb.7z $B/sg27_station9_intensity_rgb.7z &
wget $B/sg28_station4_intensity_rgb.7z $B/untermaederbrunnen_station1_xyz_intensity_rgb.7z &
wget $B/untermaederbrunnen_station3_xyz_intensity_rgb.7z &
wait
wget https://share.phys.ethz.ch/~pf/semantic3d/data/sem8_labels_training.7z
for f in *.7z; do 7z x "$f" && rm "$f"; done
echo "txt: $(ls *.txt | wc -l), labels: $(ls *.labels | wc -l)"   # mora biti 15 i 15
```

### 5. Paris-Lille-3D (~10 GB, training_10_classes)
```bash
apt-get install -y unzip
mkdir -p /workspace/data/ParisLille3D && cd /workspace/data/ParisLille3D
gdown "TVOJ_GOOGLE_DRIVE_LINK" -O ParisLille3D.zip
unzip ParisLille3D.zip && rm ParisLille3D.zip
ls **/*.ply   # Lille1_1, Lille1_2, Lille2, Paris
```

### 6. Provjera prije starta
```bash
cd /workspace/okolis-training
ls /workspace/data/Hessigheim3D/*.laz          # 3 fajla (train, val, test_GT)
ls /workspace/data/Semantic3D/*.labels | wc -l # 15
ls /workspace/runs/ptv3_big2/best.pt           # checkpoint na mjestu
grep -E "resume_from|out_dir|lr:|epochs:" config.yaml
```

### 7. Trening
```bash
cd /workspace/okolis-training
nohup python -u train.py --config config.yaml > train_big3.log 2>&1 &
tail -f train_big3.log
```
- Prvo se gradi cache (~25 min, jer je novi pod), zatim trening: 10 epoha ≈ 1.5 h.
- `tail -f` prekida se s Ctrl+C — trening nastavlja u pozadini.
- Eval svakih 5 epoha: prati **vegetation** (>0.30, ne smije se urušiti)
  i **wall** (drži 0.19+ ili raste). Best se sprema samo ako premaši 0.4163.

### 8. Nakon treninga
Skini `/workspace/runs/ptv3_big3/best.pt` na laptop (npr. kao `best_big3.pt`)
i testiraj u okolis-test:
```bash
python test_pipeline.py --weights best_big3.pt --out kod_Tina_big3.ply
```

## Aktualni parametri (Run 6 — fino podešavanje weightova)
- CLASS_WEIGHTS: wall/sidewalk 2.2 (bilo 3.5 — prejako), vegetation 0.8, ground 0.25
- lr 0.0001 (nježno), 10 epoha, resume od rekorda mIoU 0.4163
- Hessigheim w=3.0 (train + test_GT), Semantic3D 1.4, ParisLille 0.8
- out_dir: /workspace/runs/ptv3_big3, cache: /workspace/runs/ptv3_big2/scan_cache

## Povijest checkpointa
| Fajl | Model | mIoU | |
|---|---|---|---|
| best_final.pt (= ptv3_big2/best.pt) | 28M | **0.4163** | trenutni najbolji |
| best_v2fixed.pt | 7.3M | 0.33 | stara arhitektura |
| best_5ds / best_4ds / best_ptv3 | 7.3M | — | povijest |
