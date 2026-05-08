"""RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds.

Standalone implementation -- no external dependencies beyond PyTorch.
Reference: Hu et al., "RandLA-Net", CVPR 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SharedMLP(nn.Module):
    """Conv1d + BatchNorm + ReLU (operates on (B, C, N) tensors)."""

    def __init__(self, in_channels, out_channels, bn=True, activation=True):
        super().__init__()
        layers = [nn.Conv1d(in_channels, out_channels, 1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# kNN with chunked computation to avoid OOM
# ---------------------------------------------------------------------------

def chunked_knn(xyz, k, chunk_size=4096):
    """Return kNN indices (B, N, k) using chunked torch.cdist.

    Processing in chunks avoids allocating a full (B, N, N) distance matrix.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, N, k, dtype=torch.long, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(xyz[:, start:end], xyz)  # (B, chunk, N)
        _, topk_idx = dists.topk(k, dim=-1, largest=False)
        idx[:, start:end] = topk_idx
    return idx


# ---------------------------------------------------------------------------
# Random Sampling (faster than FPS for large clouds)
# ---------------------------------------------------------------------------

def random_sample(xyz, feats, n_out):
    """Randomly subsample n_out points from N points.

    Returns: xyz_sub (B, n_out, 3), feats_sub (B, n_out, C), idx (B, n_out)
    """
    B, N, _ = xyz.shape
    if n_out >= N:
        return xyz, feats, torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
    idx = torch.stack([torch.randperm(N, device=xyz.device)[:n_out] for _ in range(B)])
    idx_sorted, _ = idx.sort(dim=1)
    xyz_sub = torch.gather(xyz, 1, idx_sorted.unsqueeze(-1).expand(-1, -1, 3))
    feats_sub = torch.gather(feats, 1, idx_sorted.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
    return xyz_sub, feats_sub, idx_sorted


# ---------------------------------------------------------------------------
# Local Feature Aggregation (LFA)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learnable attention pooling over k neighbours."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2),
        )
        self.mlp = SharedMLP(in_channels, out_channels)

    def forward(self, x):
        """x: (B, N, k, C) -> (B, N, C_out)."""
        scores = self.score_fn(x)          # (B, N, k, C)
        out = (x * scores).sum(dim=2)      # (B, N, C)
        out = out.permute(0, 2, 1)         # (B, C, N)
        out = self.mlp(out)                # (B, C_out, N)
        return out.permute(0, 2, 1)        # (B, N, C_out)


class LocalFeatureAggregation(nn.Module):
    """Two rounds of local spatial encoding + attentive pooling + skip."""

    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        self.mlp_pre = SharedMLP(d_in, d_out // 2)

        # Round 1: relative pos encoding (10-dim) -> d_out//2
        self.lse1 = SharedMLP(10, d_out // 2)
        self.pool1 = AttentionPooling(d_out, d_out // 2)

        # Round 2
        self.lse2 = SharedMLP(10, d_out // 2)
        self.pool2 = AttentionPooling(d_out, d_out)

        self.shortcut = SharedMLP(d_in, d_out, activation=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    @staticmethod
    def _gather(src, idx):
        """Gather features: src (B,N,C), idx (B,N,k) -> (B,N,k,C)."""
        B, N, k = idx.shape
        C = src.shape[-1]
        idx_flat = idx.reshape(B, -1)  # (B, N*k)
        out = torch.gather(src, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
        return out.reshape(B, N, k, C)

    def _relative_pos_encoding(self, xyz, neigh_idx):
        """Compute 10-dim relative position encoding."""
        B, N, k = neigh_idx.shape
        neigh_xyz = self._gather(xyz, neigh_idx)             # (B, N, k, 3)
        center_xyz = xyz.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, 3)
        diff = neigh_xyz - center_xyz                         # (B, N, k, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)         # (B, N, k, 1)
        # [center, neighbour, diff, dist] -> 10 dims
        return torch.cat([center_xyz, neigh_xyz, diff, dist], dim=-1)

    def _encode_rpe(self, rpe, lse_module):
        """Encode relative position: (B,N,k,10) -> (B,N,k,d_out//2)."""
        B, N, k, _ = rpe.shape
        rpe_flat = rpe.reshape(B * N, k, 10).permute(0, 2, 1)  # (B*N, 10, k)
        enc = lse_module(rpe_flat)                               # (B*N, d_out//2, k)
        return enc.permute(0, 2, 1).reshape(B, N, k, -1)        # (B, N, k, d_out//2)

    def forward(self, xyz, feats, neigh_idx):
        """
        xyz:       (B, N, 3)
        feats:     (B, N, d_in)
        neigh_idx: (B, N, k)
        Returns:   (B, N, d_out)
        """
        # Pre-MLP
        f_pc = self.mlp_pre(feats.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, d_out//2)

        # Relative position encoding (shared for both rounds)
        rpe = self._relative_pos_encoding(xyz, neigh_idx)  # (B, N, k, 10)

        # Round 1
        rpe_enc1 = self._encode_rpe(rpe, self.lse1)           # (B, N, k, d_out//2)
        f_neighbours1 = self._gather(f_pc, neigh_idx)         # (B, N, k, d_out//2)
        f_concat1 = torch.cat([rpe_enc1, f_neighbours1], dim=-1)  # (B, N, k, d_out)
        f_agg1 = self.pool1(f_concat1)                        # (B, N, d_out//2)

        # Round 2
        rpe_enc2 = self._encode_rpe(rpe, self.lse2)           # (B, N, k, d_out//2)
        f_neighbours2 = self._gather(f_agg1, neigh_idx)       # (B, N, k, d_out//2)
        f_concat2 = torch.cat([rpe_enc2, f_neighbours2], dim=-1)
        f_agg2 = self.pool2(f_concat2)                        # (B, N, d_out)

        # Shortcut
        shortcut = self.shortcut(feats.permute(0, 2, 1)).permute(0, 2, 1)
        return self.lrelu(f_agg2 + shortcut)


# ---------------------------------------------------------------------------
# RandLA-Net
# ---------------------------------------------------------------------------

class RandLANet(nn.Module):
    """RandLA-Net for semantic segmentation of point clouds.

    Args:
        in_feat_dim: input feature dimension (default 5: RGB + intensity + HAG)
        num_classes: number of output classes (default 8)
        d_out:       feature dimensions at each encoding stage
        k:           number of nearest neighbours
    """

    def __init__(self, in_feat_dim=5, num_classes=8, d_out=None, k=16):
        super().__init__()
        if d_out is None:
            d_out = [32, 64, 128, 256]

        self.k = k
        self.num_classes = num_classes
        self.d_out = d_out
        self.n_layers = len(d_out)

        # Initial feature lifting (xyz concat with feats)
        d_in = 3 + in_feat_dim
        self.fc_start = SharedMLP(d_in, d_out[0])

        # Encoder: LFA at each resolution level (same in/out dims)
        self.encoders = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoders.append(LocalFeatureAggregation(d_out[i], d_out[i], k=k))

        # Dimension-lifting MLPs between encoder levels
        self.dim_up = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.dim_up.append(SharedMLP(d_out[i], d_out[i + 1]))

        # Decoder: upsample + skip connection MLPs
        self.decoder_mlps = nn.ModuleList()
        for i in range(self.n_layers - 1, 0, -1):
            self.decoder_mlps.append(SharedMLP(d_out[i] + d_out[i - 1], d_out[i - 1]))

        # Final classifier
        self.fc_end = nn.Sequential(
            SharedMLP(d_out[0], 64),
            SharedMLP(64, 32),
            nn.Dropout(0.5),
            nn.Conv1d(32, num_classes, 1),
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, N, in_feat_dim)
        Returns:
            logits:   (B, N, num_classes)
        """
        B, N, _ = xyz.shape

        # Concat xyz + features, lift to d_out[0]
        x = torch.cat([xyz, features], dim=-1)                          # (B, N, 3+d)
        x = self.fc_start(x.permute(0, 2, 1)).permute(0, 2, 1)        # (B, N, d_out[0])

        # ---- Encoder ----
        xyz_stack = [xyz]
        feat_stack = [x]

        for i, encoder in enumerate(self.encoders):
            cur_xyz = xyz_stack[-1]
            cur_feat = feat_stack[-1]
            cur_N = cur_xyz.shape[1]

            # kNN
            neigh_idx = chunked_knn(cur_xyz, self.k)
            # LFA (same in/out dims at each level)
            cur_feat = encoder(cur_xyz, cur_feat, neigh_idx)

            # Random sub-sampling (ratio 4) except last layer
            if i < self.n_layers - 1:
                # Lift dimension before downsampling: d_out[i] -> d_out[i+1]
                cur_feat = self.dim_up[i](
                    cur_feat.permute(0, 2, 1)).permute(0, 2, 1)
                n_sub = max(cur_N // 4, 1)
                sub_xyz, sub_feat, _ = random_sample(cur_xyz, cur_feat, n_sub)
                xyz_stack.append(sub_xyz)
                feat_stack.append(sub_feat)
            else:
                feat_stack[-1] = cur_feat

        # ---- Decoder (nearest-neighbour upsampling + skip) ----
        dec_feat = feat_stack[-1]
        dec_xyz = xyz_stack[-1]

        for j, mlp in enumerate(self.decoder_mlps):
            target_layer = self.n_layers - 2 - j
            target_xyz = xyz_stack[target_layer]
            skip_feat = feat_stack[target_layer]

            # Nearest-neighbour upsample
            up_feat = self._nearest_upsample(dec_xyz, target_xyz, dec_feat)
            cat_feat = torch.cat([up_feat, skip_feat], dim=-1)
            dec_feat = mlp(cat_feat.permute(0, 2, 1)).permute(0, 2, 1)
            dec_xyz = target_xyz

        # ---- Classifier ----
        logits = self.fc_end(dec_feat.permute(0, 2, 1))  # (B, C, N)
        return logits.permute(0, 2, 1)                     # (B, N, C)

    @staticmethod
    def _nearest_upsample(src_xyz, tgt_xyz, src_feat):
        """Nearest-neighbour upsample (chunked to save memory).

        src_xyz:  (B, M, 3)
        tgt_xyz:  (B, N, 3) with N > M
        src_feat: (B, M, C)
        Returns:  (B, N, C)
        """
        B, N, _ = tgt_xyz.shape
        C = src_feat.shape[-1]
        device = tgt_xyz.device
        out = torch.zeros(B, N, C, device=device, dtype=src_feat.dtype)
        chunk = 4096
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            dists = torch.cdist(tgt_xyz[:, start:end], src_xyz)  # (B, chunk, M)
            nn_idx = dists.argmin(dim=-1)                         # (B, chunk)
            out[:, start:end] = torch.gather(
                src_feat, 1, nn_idx.unsqueeze(-1).expand(-1, -1, C)
            )
        return out
