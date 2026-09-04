# E2 — fino podešavanje na kod_Tina (4 prostorna folda) — upute za pod

Trajanje: ~10 min setup + 4 × ~20 min = oko 1,5 h na RTX A6000. Ne trebaju javni skupovi.

## 1. Pod
RunPod → template **RunPod Pytorch 2.x**, GPU **RTX A6000 (48 GB)**, disk 30+ GB (Network Volume nije nužan).

## 2. Repo + paketi
```bash
pip install plyfile pyyaml scipy
cd /workspace
git clone https://github.com/Vile1278/okolis-training.git
cd okolis-training
```
Ako `finetune_folds.py` još nije u repou, uploadaj ga u `/workspace/okolis-training/` (JupyterLab drag & drop).

## 3. Upload (JupyterLab drag & drop u /workspace)
- `best_final.pt` (228 MB, iz okolis-test) → `/workspace/best_final.pt`
- `kod_Tina_train.npz` (12 MB, iz okolis-test) → `/workspace/kod_Tina_train.npz`

Provjera veličine (korumpirani upload je prošli put pukao):
```bash
ls -lh /workspace/best_final.pt /workspace/kod_Tina_train.npz   # ~228M i ~12M
python -c "import torch; ck=torch.load('/workspace/best_final.pt', map_location='cpu', weights_only=False); print(ck['epoch'], ck['miou'])"
```
Mora ispisati `1 0.4163...`.

## 4. Pokretanje
```bash
cd /workspace/okolis-training
nohup python -u finetune_folds.py \
  --npz /workspace/kod_Tina_train.npz \
  --base /workspace/best_final.pt \
  --work /workspace/e2 --folds 0 1 2 3 \
  --epochs 8 --steps 200 --lr 0.0001 > e2.log 2>&1 &
tail -f e2.log
```
Skripta za svaki fold ispiše `zero-shot: mIoU ...` pa pokrene `train.py` (log u `/workspace/e2/fold{k}/train_fold{k}.log`), pa `fine-tuned: mIoU ...`.

Ako CUDA OOM: dodaj `--crop 32768`.

## 5. Što vratiti (u `znanstveni-rad\dostava\e2\`)
- `/workspace/e2/e2_summary.json`  ← najvažnije
- `/workspace/e2/blocks.json`
- `/workspace/e2/fold*/fold*_result.json`, `fold*/train_fold*.log`, `fold*/config_fold*.yaml`
- `/workspace/e2/fold*/fold*_probs_finetuned.npz` i `fold*_probs_zeroshot.npz` (po ~2 MB, za fusion)
- `e2.log`

Najlakše: `cd /workspace && zip -r e2_rezultati.zip e2 okolis-training/e2.log -x "*/run/*" "*/cache/*" "*/data/*"` pa skini `e2_rezultati.zip`.
