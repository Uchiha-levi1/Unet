import math

import pytest
import torch

from src.losses import SegLoss, focal_loss, soft_cldice_loss, soft_dice_loss
from src.model import UNet3Plus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_CFG = {"dice": 1.0, "focal": 1.0, "cldice": 1.0, "centerline_bce": 1.0,
            "focal_alpha": 0.25, "focal_gamma": 2.0, "cldice_iters": 5}


@pytest.mark.parametrize("deep_supervision", [False, True])
@pytest.mark.parametrize("num_classes", [1, 2])
def test_output_shapes(deep_supervision, num_classes):
    model = UNet3Plus(pretrained=False, num_classes=num_classes, deep_supervision=deep_supervision).to(DEVICE)
    x = torch.randn(2, 3, 512, 512, device=DEVICE)
    model.train()
    out = model(x)
    assert out["mask"].shape == (2, num_classes, 512, 512)
    assert out["centerline"].shape == (2, 1, 512, 512)
    if deep_supervision:
        assert len(out["side"]) == 4  # D2..D5
        assert all(s.shape == (2, num_classes, 512, 512) for s in out["side"])
    else:
        assert "side" not in out
    model.eval()
    with torch.no_grad():
        assert "side" not in model(x)  # side outputs are training-only


def test_encoder_bn_stays_frozen():
    model = UNet3Plus(pretrained=False)
    model.freeze_encoder_bn = True
    model.train()
    bns = [m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    assert model.decoder.training


def _road_like_target():
    """Batch of masks with a few thick lines (so the soft skeleton is non-trivial)."""
    t = torch.zeros(2, 1, 128, 128, device=DEVICE)
    t[0, :, 40:50, :] = 1
    t[0, :, :, 70:78] = 1
    t[1, :, 10:120, 60:66] = 1
    return t


def _perfect_logits(t):
    return (t * 2 - 1) * 30.0  # sigmoid -> {~0, ~1}


@pytest.mark.parametrize("fn", [soft_dice_loss, focal_loss, soft_cldice_loss])
def test_mask_losses_finite_and_zero_when_perfect(fn):
    t = _road_like_target()
    random_loss = fn(torch.randn_like(t), t)
    perfect_loss = fn(_perfect_logits(t), t)
    for v in (random_loss, perfect_loss):
        assert v.ndim == 0 and torch.isfinite(v)
    assert perfect_loss.item() < 1e-4
    assert random_loss.item() > perfect_loss.item()


@pytest.mark.parametrize("deep_supervision", [False, True])
def test_total_loss_finite_and_zero_when_perfect(deep_supervision):
    crit = SegLoss(LOSS_CFG, centerline_pos_weight=30.0).to(DEVICE)
    mask = _road_like_target()
    cl = torch.zeros_like(mask)
    cl[0, :, 44:47, :] = 1
    side = (lambda x: [x] * 4) if deep_supervision else None
    perfect = {"mask": _perfect_logits(mask), "centerline": _perfect_logits(cl)}
    noisy = {"mask": torch.randn_like(mask), "centerline": torch.randn_like(cl)}
    if side:
        perfect["side"], noisy["side"] = side(perfect["mask"]), side(noisy["mask"])
    total_p, terms_p = crit(perfect, mask, cl)
    total_n, terms_n = crit(noisy, mask, cl)
    assert torch.isfinite(total_p) and torch.isfinite(total_n) and total_p.ndim == 0
    assert all(math.isfinite(v) for v in {**terms_p, **terms_n}.values())
    assert terms_p["centerline_bce"] < 1e-4
    assert total_p.item() < 1e-3 < total_n.item()
    assert ("side_dice" in terms_p) == deep_supervision
