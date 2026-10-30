"""Losses for a heavily imbalanced binary target (whole tumour is a few % of brain pixels).
All take raw logits (B,1,H,W) and a {0,1} target of the same shape. Overlap losses are
computed over the whole batch, so tumour-free slices do not produce 0/0."""
import torch
import torch.nn.functional as F

EPS = 1.0


def _stats(logits, target):
    p = torch.sigmoid(logits).flatten()
    t = target.flatten().float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return tp, fp, fn


def bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float())


def dice(logits, target):
    tp, fp, fn = _stats(logits, target)
    return 1 - (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def dice_bce(logits, target):
    return dice(logits, target) + bce(logits, target)


def focal(logits, target, gamma=2.0, alpha=0.75):
    """Lin et al. 2017; alpha weights the rare positive class."""
    t = target.float()
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.exp(-ce)
    a = alpha * t + (1 - alpha) * (1 - t)
    # scaled so its magnitude is comparable to BCE at initialisation
    return (a * (1 - pt) ** gamma * ce).mean() * 10


def tversky_index(logits, target, alpha=0.3, beta=0.7):
    tp, fp, fn = _stats(logits, target)
    return (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)


def tversky(logits, target, alpha=0.3, beta=0.7):
    """Salehi et al. 2017: beta > alpha penalises missed tumour more than false alarms."""
    return 1 - tversky_index(logits, target, alpha, beta)


def focal_tversky(logits, target, alpha=0.3, beta=0.7, gamma=0.75):
    """Abraham & Khan 2019."""
    return (1 - tversky_index(logits, target, alpha, beta)) ** gamma


LOSSES = {"bce": bce, "dice": dice, "dice_bce": dice_bce, "focal": focal,
          "tversky": tversky, "focal_tversky": focal_tversky}
