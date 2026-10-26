"""Per-image evaluation metrics on binary (thresholded) masks."""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def dice_iou(pred, gt):
    """NaN when both are empty (class absent and not predicted), so it does not inflate the average."""
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    if psum + gsum == 0:
        return np.nan, np.nan
    return 2 * inter / (psum + gsum), inter / (psum + gsum - inter)


def hard_cldice(pred, gt, gt_skel=None):
    """clDice on binary masks: harmonic mean of Tprec = |S(P)∩G|/|S(P)| and Tsens = |S_G∩P|/|S_G|.
    gt_skel: given GT skeleton (the stored 1 px centerline); default = skeleton of the GT mask.
    Returns (cldice, tprec, tsens); Tsens is also the share of the GT centerline covered by the prediction."""
    sp = skeletonize(pred)
    sg = skeletonize(gt) if gt_skel is None else gt_skel
    if sp.sum() == 0 and sg.sum() == 0:
        return np.nan, np.nan, np.nan
    tprec = np.logical_and(sp, gt).sum() / sp.sum() if sp.sum() else 0.0
    tsens = np.logical_and(sg, pred).sum() / sg.sum() if sg.sum() else 0.0
    cldice = 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)
    return cldice, tprec, tsens


def num_components(mask):
    return int(label(mask, connectivity=2).max())  # 8-connectivity


def image_metrics(pred, gt, classes, skels):
    """pred, gt: (C,H,W) bool in `classes` order; skels: {class: GT skeleton (H,W) bool} for clDice classes."""
    m = {}
    for i, c in enumerate(classes):
        m[f"dice/{c}"], m[f"iou/{c}"] = dice_iou(pred[i], gt[i])
    for c, skel in skels.items():
        i = classes.index(c)
        m[f"cldice/{c}"], m[f"tprec/{c}"], m[f"tsens/{c}"] = hard_cldice(pred[i], gt[i], skel)
    if "strap" in classes:
        i = classes.index("strap")
        m["cc_pred"], m["cc_gt"] = num_components(pred[i]), num_components(gt[i])
        m["cc_abs_diff"] = abs(m["cc_pred"] - m["cc_gt"])
        if "strap" in skels:
            m["strap_cl_coverage"] = m["tsens/strap"]
    return {k: float(v) for k, v in m.items()}


def average(metric_list):
    """Mean over images, ignoring NaN (metric undefined for that image)."""
    out = {}
    for k in metric_list[0]:
        v = np.array([m[k] for m in metric_list], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    return out
