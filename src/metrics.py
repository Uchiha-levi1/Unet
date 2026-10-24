"""Per-image evaluation metrics on binary (thresholded) masks."""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def dice_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    union = psum + gsum - inter
    if psum + gsum == 0:  # both empty: perfect
        return 1.0, 1.0
    return 2 * inter / (psum + gsum), inter / union


def hard_cldice(pred, gt):
    """clDice on binary masks: harmonic mean of Tprec = |S(P)∩G|/|S(P)| and Tsens = |S(G)∩P|/|S(G)|."""
    sp, sg = skeletonize(pred), skeletonize(gt)
    if sp.sum() == 0 and sg.sum() == 0:
        return 1.0
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = np.logical_and(sp, gt).sum() / sp.sum()
    tsens = np.logical_and(sg, pred).sum() / sg.sum()
    return 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)


def num_components(mask):
    return int(label(mask, connectivity=2).max())  # 8-connectivity


def image_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    dice, iou = dice_iou(pred, gt)
    cc_pred, cc_gt = num_components(pred), num_components(gt)
    return {"dice": float(dice), "iou": float(iou), "cldice": float(hard_cldice(pred, gt)),
            "cc_pred": cc_pred, "cc_gt": cc_gt, "cc_abs_diff": abs(cc_pred - cc_gt)}


def average(metric_list):
    return {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}
