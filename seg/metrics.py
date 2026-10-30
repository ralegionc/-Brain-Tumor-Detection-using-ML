"""Volume-level segmentation metrics and false-positive/negative error patterns.
Masks are boolean arrays (Z,H,W); spacing is (dz, dy, dx) in mm."""
import numpy as np
from scipy import ndimage


def overlap(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = pred.size - tp - fp - fn
    if tp + fp + fn == 0:  # both empty: perfect agreement
        return dict(dice=1.0, iou=1.0, sensitivity=1.0, precision=1.0, specificity=1.0)
    return dict(
        dice=2 * tp / (2 * tp + fp + fn),
        iou=tp / (tp + fp + fn),
        sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        precision=tp / (tp + fp) if tp + fp else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
    )


def _surface(m):
    return np.logical_xor(m, ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(m.ndim, 1)))


def surface_distances(a, b, spacing):
    """Distances (mm) from each surface voxel of a to the nearest surface voxel of b."""
    sa, sb = _surface(a), _surface(b)
    dt = ndimage.distance_transform_edt(~sb, sampling=spacing)
    return dt[sa]


def hausdorff(pred, gt, spacing, percentile=95):
    """Symmetric Hausdorff distance (max, or a percentile of the pooled surface distances).
    Returns nan when exactly one mask is empty and 0 when both are."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan
    d = np.concatenate([surface_distances(pred, gt, spacing), surface_distances(gt, pred, spacing)])
    return float(np.max(d) if percentile == 100 else np.percentile(d, percentile))


def _crop(pred, gt, pad=2):
    """Bounding box of pred | gt plus a margin. Every surface and every error voxel lies
    inside it, so distance transforms and component labels computed on the crop are exact."""
    u = pred | gt
    if not u.any():
        return pred, gt
    idx = [np.nonzero(u.any(axis=tuple(j for j in range(u.ndim) if j != i)))[0] for i in range(u.ndim)]
    box = tuple(slice(max(i[0] - pad, 0), i[-1] + pad + 1) for i in idx)
    return pred[box], gt[box]


def volume_metrics(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    m = overlap(pred, gt)  # on the full volume: specificity depends on the background count
    p, g = _crop(pred, gt)
    if p.any() and g.any():
        d = np.concatenate([surface_distances(p, g, spacing), surface_distances(g, p, spacing)])
        m["hd95"], m["hd"] = float(np.percentile(d, 95)), float(np.max(d))
    else:
        m["hd95"] = m["hd"] = hausdorff(p, g, spacing)
    return m


def error_patterns(pred, gt, spacing, adjacent_mm=5.0, min_ml=0.1):
    """Split errors into the patterns a reader would flag:
    - boundary FP: false-positive voxels within adjacent_mm of the true tumour (over-segmentation)
    - distant FP: false-positive voxels further away (normal anatomy called tumour)
    - FP components: 3D connected predicted blobs that touch no true tumour
    - boundary FN: missed tumour voxels within adjacent_mm of the predicted region (under-segmentation)
    - missed lesions: true tumour components the prediction does not touch at all
    The *_big counts only include components of at least min_ml. Volumes are in mL."""
    pred, gt = _crop(pred.astype(bool), gt.astype(bool))
    vox_ml = float(np.prod(spacing)) / 1000.0
    st = ndimage.generate_binary_structure(3, 3)
    fp, fn = pred & ~gt, gt & ~pred
    d_gt = ndimage.distance_transform_edt(~gt, sampling=spacing) if gt.any() else np.full(gt.shape, np.inf)
    d_pr = ndimage.distance_transform_edt(~pred, sampling=spacing) if pred.any() else np.full(gt.shape, np.inf)
    lab_p, n_p = ndimage.label(pred, st)
    lab_g, n_g = ndimage.label(gt, st)
    hit_p = set(np.unique(lab_p[gt])) - {0}
    hit_g = set(np.unique(lab_g[pred])) - {0}
    # most ground-truth components are resampling specks (61% are < 0.1 mL but hold 0.03%
    # of tumour volume), so lesion counts are also reported for components >= min_ml
    big_p = {i for i, n in enumerate(np.bincount(lab_p.ravel()) * vox_ml) if i and n >= min_ml}
    big_g = {i for i, n in enumerate(np.bincount(lab_g.ravel()) * vox_ml) if i and n >= min_ml}
    return dict(
        fp_components_big=len(big_p - hit_p),
        missed_lesions_big=len(big_g - hit_g),
        gt_lesions_big=len(big_g),
        fp_boundary_ml=(fp & (d_gt <= adjacent_mm)).sum() * vox_ml,
        fp_distant_ml=(fp & (d_gt > adjacent_mm)).sum() * vox_ml,
        fp_components=n_p - len(hit_p),
        fn_boundary_ml=(fn & (d_pr <= adjacent_mm)).sum() * vox_ml,
        fn_distant_ml=(fn & (d_pr > adjacent_mm)).sum() * vox_ml,
        missed_lesions=n_g - len(hit_g),
        gt_lesions=n_g,
        gt_ml=gt.sum() * vox_ml,
        pred_ml=pred.sum() * vox_ml,
    )
