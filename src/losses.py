"""Segmentation losses. All inputs are raw logits; everything is computed in float32."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_per_channel(logits, target, smooth=1.0):
    """1 - Dice per channel, computed over the whole batch (so a class absent from one image is harmless). (C,)"""
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1 - (2 * inter + smooth) / (denom + smooth)


def soft_dice_loss(logits, target, smooth=1.0):
    return dice_per_channel(logits, target, smooth).mean()


def focal_per_channel(logits, target, gamma=2.0, alpha=None):
    """Sigmoid focal loss (Lin et al., 2017), mean over all pixels of each channel. (C,)"""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean(dim=(0, 2, 3))


def focal_loss(logits, target, gamma=2.0, alpha=None):
    return focal_per_channel(logits, target, gamma, alpha).mean()


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


def soft_cldice_loss(logits, target, iterations=5, smooth=1.0, skel_target=None):
    """1 - clDice: harmonic mean of topology precision (pred skeleton inside GT mask)
    and topology sensitivity (GT skeleton inside predicted mask).
    skel_target: a given GT skeleton (e.g. the stored 1 px centerline); default = soft skeleton of the GT mask."""
    p = torch.sigmoid(logits)
    skel_p = soft_skeleton(p, iterations)
    skel_t = soft_skeleton(target, iterations) if skel_target is None else skel_target
    dims = (0, 2, 3)
    tprec = ((skel_p * target).sum(dims) + smooth) / (skel_p.sum(dims) + smooth)
    tsens = ((skel_t * p).sum(dims) + smooth) / (skel_t.sum(dims) + smooth)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


class SegLoss(nn.Module):
    """Total = Σ_c λ_c·(Dice_c + Focal_c)                     (every mask channel)
             + Σ_k w_k·clDice_k                                (configured classes, final output only)
             + λ_cl·weighted BCE                               (strap centerline head, 3 px target)
             + Σ_side Σ_c λ_c·(Dice_c + Focal_c)               (deep supervision: side classes only)

    Targets come as a dict: "mask" (B,C,H,W) plus the centerline maps named in the config."""

    CENTERLINE_TARGET = "strap_cl_3px"

    def __init__(self, loss_cfg, classes, centerline_pos_weight, side_classes=()):
        super().__init__()
        self.classes = list(classes)
        self.gamma, self.alpha = loss_cfg["focal_gamma"], loss_cfg["focal_alpha"]
        self.register_buffer("lam", torch.tensor([float(loss_cfg["class_weights"][c]) for c in self.classes]))
        # clDice terms with weight 0 are skipped entirely (the soft skeleton is comparatively expensive).
        self.cldice = [(self.classes.index(c), c, float(v["weight"]), v["iters"], v["skeleton"])
                       for c, v in loss_cfg["cldice"].items() if v["weight"] > 0]
        self.w_cl = loss_cfg["centerline_bce"]
        self.register_buffer("pos_weight", torch.tensor([float(centerline_pos_weight)]))
        self.side_idx = [self.classes.index(c) for c in side_classes]

    def forward(self, out, targets):
        logits, mask = out["mask"].float(), targets["mask"]
        dice = dice_per_channel(logits, mask)
        focal = focal_per_channel(logits, mask, self.gamma, self.alpha)
        terms = {**{f"dice/{c}": dice[i] for i, c in enumerate(self.classes)},
                 **{f"focal/{c}": focal[i] for i, c in enumerate(self.classes)}}
        total = (self.lam * (dice + focal)).sum()

        for i, name, w, iters, skel_key in self.cldice:
            v = soft_cldice_loss(logits[:, i:i + 1], mask[:, i:i + 1], iters, skel_target=targets[skel_key])
            terms[f"cldice/{name}"] = v
            total = total + w * v

        bce = F.binary_cross_entropy_with_logits(out["centerline"].float(), targets[self.CENTERLINE_TARGET],
                                                 pos_weight=self.pos_weight)
        terms["centerline_bce"] = bce
        total = total + self.w_cl * bce

        if "side" in out:  # deep supervision: Dice + focal on the side classes only, never clDice
            side_mask, lam = mask[:, self.side_idx], self.lam[self.side_idx]
            side = 0.0
            for s in out["side"]:
                s = s.float()
                side = side + (lam * (dice_per_channel(s, side_mask)
                                      + focal_per_channel(s, side_mask, self.gamma, self.alpha))).sum()
            terms["side"] = side
            total = total + side

        terms["total"] = total
        return total, {k: v.detach() for k, v in terms.items()}
