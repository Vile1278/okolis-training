"""Point Transformer V3 for outdoor point cloud semantic segmentation.

Standalone implementation — no external dependencies beyond PyTorch.
Reference: Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024.

Key ideas:
  - Serialize unordered points via z-order (Morton) curves
  - Windowed self-attention along the serialized order
  - GridPool / GridUnpool for multi-scale encoder-decoder
  - 4-stage architecture with skip connections

Backward-compatible: `RandLANet = PointTransformerV3` alias so existing
train.py / inference code keeps working without changes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Serialization: Z-order (Morton) curve
# ============================================================================

def _interlace_bits(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Interleave lower 21 bits of x, y, z into a 63-bit Morton code.

    Works on integer tensors. Output dtype is int64.
    """
    def spread(v: torch.Tensor) -> torch.Tensor:
        # Spread 21 bits of v across 63 bits (every 3rd position)
        v = v.long() & 0x1FFFFF  # clamp to 21 bits
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8))  & 0x100F00F00F00F00F
        v = (v | (v << 4))  & 0x10C30C30C30C30C3
        v = (v | (v << 2))  & 0x1249249249249249
        return v

    return spread(x) | (spread(y) << 1) | (spread(z) << 2)


def serialize_points(xyz: torch.Tensor, grid_size: float = 0.04) -> torch.Tensor:
    """Compute z-order keys for point serialization.

    Args:
        xyz: (B, N, 3) point coordinates
        grid_size: voxel resolution for quantization

    Returns:
        order: (B, N) indices that sort points along z-order curve
    """
    B, N, _ = xyz.shape
    # Quantize to grid
    coords = torch.floor(xyz / grid_size).long()
    # Shift to non-negative
    mins = coords.min(dim=1, keepdim=True).values
    coords = coords - mins

    codes = _interlace_bits(coords[..., 0], coords[..., 1], coords[..., 2])
    order = codes.argsort(dim=1)
    return order


def reorder(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Reorder tensor x (B, N, C) according to order (B, N)."""
    B, N, C = x.shape
    idx = order.unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(x, 1, idx)


def unreorder(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Inverse of reorder: scatter back to original positions."""
    B, N, C = x.shape
    idx = order.unsqueeze(-1).expand(-1, -1, C)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, x)
    return out


# ============================================================================
# Windowed Multi-Head Self-Attention
# ============================================================================

class WindowedAttention(nn.Module):
    """Multi-head self-attention within windows of serialized points."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 256,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) — already serialized
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        W = self.window_size

        # Pad N to multiple of W
        pad = (W - N % W) % W
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # pad N dimension
        Np = N + pad
        nW = Np // W  # number of windows

        # Reshape into windows: (B * nW, W, C)
        x = x.reshape(B, nW, W, C).reshape(B * nW, W, C)

        # QKV
        qkv = self.qkv(x).reshape(B * nW, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, heads, W, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * nW, W, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back and remove padding
        out = out.reshape(B, nW, W, C).reshape(B, Np, C)
        if pad > 0:
            out = out[:, :N]

        return out


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attention → LN → MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 256,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(
            dim, num_heads, window_size=window_size,
            attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Grid Pooling / Unpooling (multi-scale)
# ============================================================================

class GridPool(nn.Module):
    """Voxel-based downsampling: average-pool features within grid cells.

    Maps N points → M voxels (M < N). Returns pooled features, new xyz,
    and the cluster assignments needed for unpooling.
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: float = 0.08):
        super().__init__()
        self.grid_size = grid_size
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
            feats: (B, N, C_in)
        Returns:
            xyz_pooled: (B, M, 3)
            feats_pooled: (B, M, C_out)
            cluster: (B, N) — which voxel each point belongs to
            M: int — number of voxels (max across batch)
        """
        B, N, C = feats.shape
        g = self.grid_size
        device = xyz.device

        all_xyz = []
        all_feats = []
        all_cluster = []
        max_M = 0

        for b in range(B):
            coords = torch.floor(xyz[b] / g).long()  # (N, 3)
            mins = coords.min(dim=0).values
            coords = coords - mins

            # Hash to flat index
            dims = coords.max(dim=0).values + 1
            keys = (coords[:, 0] * dims[1] * dims[2]
                    + coords[:, 1] * dims[2]
                    + coords[:, 2])

            unique_keys, cluster = torch.unique(keys, return_inverse=True)
            M = len(unique_keys)
            max_M = max(max_M, M)

            # Scatter-mean for xyz and feats
            xyz_sum = torch.zeros(M, 3, device=device, dtype=xyz.dtype)
            feat_sum = torch.zeros(M, C, device=device, dtype=feats.dtype)
            count = torch.zeros(M, device=device, dtype=feats.dtype)

            xyz_sum.scatter_add_(0, cluster.unsqueeze(-1).expand(-1, 3), xyz[b])
            feat_sum.scatter_add_(0, cluster.unsqueeze(-1).expand(-1, C), feats[b])
            count.scatter_add_(0, cluster, torch.ones(N, device=device, dtype=feats.dtype))
            count = count.clamp(min=1)

            xyz_mean = xyz_sum / count.unsqueeze(-1)
            feat_mean = feat_sum / count.unsqueeze(-1)

            all_xyz.append(xyz_mean)
            all_feats.append(feat_mean)
            all_cluster.append(cluster)

        # Pad to max_M across batch
        xyz_pooled = torch.zeros(B, max_M, 3, device=device, dtype=xyz.dtype)
        feats_pooled = torch.zeros(B, max_M, C, device=device, dtype=feats.dtype)
        cluster_out = torch.zeros(B, N, device=device, dtype=torch.long)

        for b in range(B):
            M = len(all_xyz[b])
            xyz_pooled[b, :M] = all_xyz[b]
            feats_pooled[b, :M] = all_feats[b]
            cluster_out[b] = all_cluster[b]

        feats_pooled = self.norm(self.linear(feats_pooled))
        return xyz_pooled, feats_pooled, cluster_out, max_M


class GridUnpool(nn.Module):
    """Upsample by scattering voxel features back to original points.

    Uses cluster assignments from GridPool + skip connection from encoder.
    """

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim + skip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feats: torch.Tensor, cluster: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, M, C_in) — decoder features at coarse level
            cluster: (B, N) — cluster assignments from GridPool
            skip: (B, N, C_skip) — encoder skip features at fine level
        Returns:
            (B, N, C_out)
        """
        B, N = cluster.shape
        C_in = feats.shape[-1]

        # Gather: scatter coarse features to fine points
        idx = cluster.unsqueeze(-1).expand(-1, -1, C_in)
        upsampled = torch.gather(feats, 1, idx)  # (B, N, C_in)

        cat = torch.cat([upsampled, skip], dim=-1)
        return self.norm(self.linear(cat))


# ============================================================================
# Point Transformer V3
# ============================================================================

class PointTransformerV3(nn.Module):
    """Point Transformer V3 for semantic segmentation.

    4-stage encoder-decoder with serialized windowed attention and
    grid-based pooling/unpooling.

    Args:
        in_feat_dim: input per-point feature dimension (5 = RGB + intensity + HAG)
        num_classes: number of output semantic classes
        dims: channel dimensions at each encoder stage
        num_heads: attention heads at each stage
        depths: number of transformer blocks at each stage
        window_size: window size for serialized attention
        grid_sizes: voxel sizes for GridPool between stages (len = n_stages - 1)
        drop: dropout rate
        serialize_grid: voxel size for z-order serialization
    """

    def __init__(self, in_feat_dim: int = 5, num_classes: int = 8,
                 dims: tuple = (48, 96, 192, 384),
                 num_heads: tuple = (3, 6, 12, 24),
                 depths: tuple = (2, 2, 6, 2),
                 window_size: int = 256,
                 grid_sizes: tuple = (0.08, 0.16, 0.32),
                 drop: float = 0.0,
                 serialize_grid: float = 0.04):
        super().__init__()
        self.num_classes = num_classes
        self.n_stages = len(dims)
        self.serialize_grid = serialize_grid

        assert len(dims) == len(num_heads) == len(depths)
        assert len(grid_sizes) == len(dims) - 1

        # ── Input embedding ──────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(3 + in_feat_dim, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.GELU(),
            nn.Linear(dims[0], dims[0]),
            nn.LayerNorm(dims[0]),
        )

        # ── Encoder stages ───────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        for i in range(self.n_stages):
            stage = nn.Sequential(*[
                TransformerBlock(dims[i], num_heads[i],
                                window_size=window_size,
                                mlp_ratio=4.0, drop=drop)
                for _ in range(depths[i])
            ])
            self.encoder_blocks.append(stage)

        # ── Grid pooling (between encoder stages) ────────────────────
        self.pools = nn.ModuleList()
        for i in range(self.n_stages - 1):
            self.pools.append(GridPool(dims[i], dims[i + 1], grid_sizes[i]))

        # ── Decoder (unpool + skip + transformer block) ──────────────
        self.unpools = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.n_stages - 2, -1, -1):
            self.unpools.append(GridUnpool(dims[i + 1], dims[i], dims[i]))
            self.decoder_blocks.append(
                TransformerBlock(dims[i], num_heads[i],
                                window_size=window_size,
                                mlp_ratio=4.0, drop=drop))

        # ── Classification head ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(dims[0], dims[0]),
            nn.LayerNorm(dims[0]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(dims[0], num_classes),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz:      (B, N, 3) point coordinates
            features: (B, N, in_feat_dim) per-point features
        Returns:
            logits:   (B, N, num_classes)
        """
        B, N, _ = xyz.shape

        # ── Serialize points via z-order curve ───────────────────────
        order = serialize_points(xyz, grid_size=self.serialize_grid)
        xyz_s = reorder(xyz, order)
        feat_s = reorder(features, order)

        # ── Input projection ─────────────────────────────────────────
        x = torch.cat([xyz_s, feat_s], dim=-1)  # (B, N, 3+d)
        x = self.input_proj(x)                   # (B, N, dims[0])

        # ── Encoder ──────────────────────────────────────────────────
        enc_feats = []    # skip connections
        enc_xyz = []      # xyz at each scale
        enc_clusters = [] # cluster assignments for unpooling

        cur_xyz = xyz_s
        cur_feat = x

        for i in range(self.n_stages):
            cur_feat = self.encoder_blocks[i](cur_feat)
            enc_feats.append(cur_feat)
            enc_xyz.append(cur_xyz)

            if i < self.n_stages - 1:
                cur_xyz, cur_feat, cluster, _ = self.pools[i](cur_xyz, cur_feat)
                enc_clusters.append(cluster)

                # Re-serialize at new scale
                new_order = serialize_points(cur_xyz, grid_size=self.serialize_grid)
                cur_xyz = reorder(cur_xyz, new_order)
                cur_feat = reorder(cur_feat, new_order)

        # ── Decoder ──────────────────────────────────────────────────
        dec_feat = cur_feat

        for j in range(self.n_stages - 1):
            # j=0 → go from stage (n-1) to stage (n-2), etc.
            enc_idx = self.n_stages - 2 - j
            cluster = enc_clusters[enc_idx]
            skip = enc_feats[enc_idx]

            dec_feat = self.unpools[j](dec_feat, cluster, skip)
            dec_feat = self.decoder_blocks[j](dec_feat)

        # ── Unsort back to original point order ──────────────────────
        dec_feat = unreorder(dec_feat, order)

        # ── Classify ─────────────────────────────────────────────────
        logits = self.head(dec_feat)  # (B, N, num_classes)
        return logits


# ============================================================================
# Backward-compatible alias
# ============================================================================

RandLANet = PointTransformerV3
