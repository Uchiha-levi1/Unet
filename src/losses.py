"""Segmentation losses. All inputs are raw logits; everything is computed in float32."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, smooth=1.0):
    """1 - Dice, per channel over the whole batch, averaged over channels."""
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    """Sigmoid focal loss (Lin et al., 2017), mean over pixels."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean()


# --- Soft skeleton (Shit et al., CVPR 2021) ---------------------------------------------------
# Morphology on soft (probability) maps: erosion = min-pool, dilation = max-pool, both done
# with max_pool2d (min-pool = -maxpool(-x)). Erosion uses a plus-shaped 3x3 element (min over a
# 3x1 and a 1x3 window). The skeleton is accumulated like the morphological skeleton: at each
# erosion step, the pixels removed by an opening (x - open(x)) belong to the skeleton.

def soft_erode(x):
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iterations):
    skel = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        # Add only the part of delta not already in skel (soft "union" without double counting).
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits, target, iterations=5, smooth=1.0):
    """1 - clDice: harmonic mean of topology precision (pred skeleton inside GT mask)
    and topology sensitivity (GT skeleton inside predicted mask)."""
    p = torch.sigmoid(logits)
    skel_p = soft_skeleton(p, iterations)
    skel_t = soft_skeleton(target, iterations)
    dims = (0, 2, 3)
    tprec = ((skel_p * target).sum(dims) + smooth) / (skel_p.sum(dims) + smooth)
    tsens = ((skel_t * p).sum(dims) + smooth) / (skel_t.sum(dims) + smooth)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


class SegLoss(nn.Module):
    """Total = λ_dice·Dice + λ_focal·Focal + λ_cldice·clDice (final mask output)
             + λ_cl·weighted BCE (centerline)
             + Σ_side (λ_dice·Dice + λ_focal·Focal) when deep-supervision outputs are present."""

    def __init__(self, loss_cfg, centerline_pos_weight):
        super().__init__()
        self.w = {k: loss_cfg[k] for k in ("dice", "focal", "cldice", "centerline_bce")}
        self.alpha, self.gamma = loss_cfg["focal_alpha"], loss_cfg["focal_gamma"]
        self.iters = loss_cfg["cldice_iters"]
        self.register_buffer("pos_weight", torch.tensor([float(centerline_pos_weight)]))

    def forward(self, out, mask, centerline):
        logits = out["mask"].float()
        terms = {
            "dice": soft_dice_loss(logits, mask),
            "focal": focal_loss(logits, mask, self.gamma, self.alpha),
            "centerline_bce": F.binary_cross_entropy_with_logits(
                out["centerline"].float(), centerline, pos_weight=self.pos_weight),
        }
        # Skip the (comparatively expensive) soft skeleton entirely in the ablation.
        terms["cldice"] = (soft_cldice_loss(logits, mask, self.iters) if self.w["cldice"] > 0
                           else logits.new_zeros(()))
        total = sum(self.w[k] * v for k, v in terms.items())

        if "side" in out:  # deep supervision: Dice + focal only, never clDice
            side_dice = side_focal = 0.0
            for s in out["side"]:
                s = s.float()
                side_dice = side_dice + soft_dice_loss(s, mask)
                side_focal = side_focal + focal_loss(s, mask, self.gamma, self.alpha)
            terms["side_dice"], terms["side_focal"] = side_dice, side_focal
            total = total + self.w["dice"] * side_dice + self.w["focal"] * side_focal

        terms["total"] = total
        return total, {k: float(v.detach()) for k, v in terms.items()}
