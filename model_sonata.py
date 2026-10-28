"""Sonata (facebook/sonata, CVPR 2025) pretrenirani PTv3 encoder +
segmentacijska glava za Okoliš AI.

Sonata je self-supervised pretreniran Point Transformer V3 ENCODER
(bez decodera) treniran na milijunima 3D scena. Mi na njega stavljamo
laganu MLP glavu i fino treniramo na našim datasetima.

NAPOMENA O LICENCI: Sonata TEŽINE su CC-BY-NC 4.0 (nekomercijalno).
Za znanstveni rad i studentski projekt OK; za komercijalni proizvod
treba zamjena.

Instalacija na podu (uz postojeće pakete):
    pip install spconv-cu121 huggingface_hub timm   # cu121/cu124 po "nvcc --version"
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    pip install git+https://github.com/facebookresearch/sonata.git
    # flash-attention NIJE obavezan — koristimo enable_flash=False

Ulaz (iz train.py, feat_mode="sonata11"):
    xyz   (B, N, 3)  — centrirane koordinate cropa
    feats (B, N, 11) — [R,G,B, intensity, HAG, nx,ny,nz, lin,plan,sph]

Encoder dobiva što Sonata očekuje: coord + color + normal (9 dim).
Dodatne značajke [intensity, HAG, lin, plan, sph] ulaze u GLAVU,
konkatenirane s encoder featurima — tako tensor značajke (Wang et al.
2022) i HAG ne diraju pretrenirane težine.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SonataSeg(nn.Module):
    EXTRA_IDX = [3, 4, 8, 9, 10]   # intensity, HAG, lin, plan, sph
    COLOR_IDX = [0, 1, 2]
    NORMAL_IDX = [5, 6, 7]

    def __init__(self, num_classes=8, ckpt=None, enable_flash=False,
                 head_hidden=256, grid_size=0.02, drop=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        import sonata  # pip install git+https://github.com/facebookresearch/sonata.git
        custom = {}
        if not enable_flash:
            # Bez FlashAttentiona: README recept — veći patch, flash off
            custom = dict(enc_patch_size=[1024] * 5, enable_flash=False)
        if ckpt:
            self.encoder = sonata.model.load(ckpt, custom_config=custom)
        else:
            self.encoder = sonata.model.load(
                "sonata", repo_id="facebook/sonata", custom_config=custom)

        # Glava: [encoder feat (dim ovisi o upcastu) + 5 extra] → klase.
        # Dimenzija encoder izlaza ovisi o konfiguraciji (upcast konkatenira
        # zadnje 3 razine, npr. 128+256+512), pa je SONDIRAMO probnim
        # forwardom na CPU — glava mora biti materijalizirana PRIJE nego
        # train.py izgradi optimizer i EMA state.
        enc_dim = self._probe_enc_dim()
        head_in = enc_dim + len(self.EXTRA_IDX)
        print(f"  [sonata] encoder feature dim: {enc_dim} "
              f"(+{len(self.EXTRA_IDX)} extra = {head_in} u glavu)")
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, num_classes),
        )

    def _probe_enc_dim(self, n=4096):
        """Mini probni forward (CPU ili CUDA) da se utvrdi izlazna dimenzija."""
        dev = next(self.encoder.parameters()).device
        g = torch.Generator().manual_seed(0)
        coord = torch.rand((n, 3), generator=g).to(dev) * 4.0
        color = torch.zeros((n, 3), device=dev)
        normal = torch.zeros((n, 3), device=dev)
        normal[:, 2] = 1.0
        batch_idx = torch.zeros(n, dtype=torch.long, device=dev)
        with torch.no_grad():
            feat = self._encode(coord, color, normal, batch_idx)
        return feat.shape[1]

    # ------------------------------------------------------------------
    def _encode(self, coord, color, normal, batch_idx):
        """Pokreni Sonata encoder i vrati per-point feature (M, C)."""
        # Pointcept Point dict. feat = (coord, color, normal) kao u
        # službenom Sonata inference receptu (feat_keys).
        offset = torch.cumsum(torch.bincount(batch_idx), dim=0).int()
        grid_coord = torch.floor(
            (coord - coord.min(dim=0, keepdim=True).values) / self.grid_size
        ).int()
        data = {
            "coord": coord,
            "grid_coord": grid_coord,
            "feat": torch.cat([coord, color, normal], dim=1),
            "offset": offset,
            "batch": batch_idx,
            "grid_size": self.grid_size,
        }
        point = self.encoder(data)

        # Službeni upcast recept iz Sonata README-a: prve 2 razine
        # konkateniraju feature djeteta u roditelja, ostale samo mapiraju.
        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point.feat  # (M, C) — M = broj točaka nakon internog gridanja

        # Napomena: Sonata interno NE radi dodatni GridSample (mi dajemo
        # već voxelizirane 2cm točke), pa je M == broj ulaznih točaka.

    # ------------------------------------------------------------------
    def forward(self, xyz, feats):
        """Isti interface kao naš PointTransformerV3: (B,N,3),(B,N,11) → (B,N,K)."""
        B, N, _ = xyz.shape
        coord = xyz.reshape(B * N, 3)
        f = feats.reshape(B * N, feats.shape[-1])
        batch_idx = torch.arange(B, device=xyz.device).repeat_interleave(N)

        color = f[:, self.COLOR_IDX] * 2.0 - 1.0   # NormalizeColor parity
        normal = f[:, self.NORMAL_IDX]
        extra = f[:, self.EXTRA_IDX]

        enc = self._encode(coord, color, normal, batch_idx)
        if enc.shape[0] != coord.shape[0]:
            raise RuntimeError(
                f"Sonata vratila {enc.shape[0]} featura za {coord.shape[0]} "
                f"točaka — encoder je interno re-gridao. Provjeri grid_size.")

        logits = self.head(torch.cat([enc, extra], dim=1))
        return logits.reshape(B, N, self.num_classes)
