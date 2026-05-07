"""Lovasz-Softmax and weighted cross-entropy losses."""
import torch
import torch.nn.functional as F


def class_weighted_ce(logits, labels, weights=None, ignore_index=0):
    """Cross-entropy loss with optional class weights.

    Args:
        logits: (B, N, C) raw predictions
        labels: (B, N) integer labels
        weights: optional list of C floats
        ignore_index: label to ignore
    """
    B, N, C = logits.shape
    logits = logits.reshape(-1, C)
    labels = labels.reshape(-1)
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=logits.device)
    else:
        w = None
    return F.cross_entropy(logits, labels, weight=w, ignore_index=ignore_index)


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszSoftmax(torch.nn.Module):
    """Lovasz-Softmax loss for multi-class segmentation."""

    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, N, C) raw predictions
            labels: (B, N) integer labels
        """
        B, N, C = logits.shape
        probs = F.softmax(logits, dim=-1)
        loss = 0.0
        count = 0
        for b in range(B):
            p = probs[b]   # (N, C)
            t = labels[b]  # (N,)
            valid = t != self.ignore
            if valid.sum() < 2:
                continue
            p = p[valid]
            t = t[valid]
            for c in range(1, C):
                fg = (t == c).float()
                if fg.sum() == 0 and (1 - fg).sum() == 0:
                    continue
                errors = (fg - p[:, c]).abs()
                errors_sorted, perm = torch.sort(errors, descending=True)
                fg_sorted = fg[perm]
                grad = lovasz_grad(fg_sorted)
                loss += torch.dot(errors_sorted, grad)
                count += 1
        return loss / max(count, 1)
