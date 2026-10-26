import math

import pytest
import torch

from src.losses import SegLoss, focal_loss, soft_cldice_loss, soft_dice_loss
from src.model import UNet3Plus
from src.train import lr_factor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 768
CLASSES = ["strap", "cargo_cover", "cord_or_tie", "hardware"]
SIDE = ["cargo_cover"]
LOSS_CFG = {"class_weights": {"strap": 1.0, "cargo_cover": 0.25, "cord_or_tie": 1.0, "hardware": 1.0},
            "focal_gamma": 2.0, "focal_alpha": None, "centerline_bce": 1.0,
            "cldice": {"strap": {"weight": 2.0, "iters": 5, "skeleton": "strap_cl_1px"},
                       "cord_or_tie": {"weight": 2.0, "iters": 3, "skeleton": "cord_cl_1px"}}}


@pytest.mark.parametrize("deep_supervision", [False, True])
@pytest.mark.parametrize("num_classes", [1, 10])
def test_output_shapes(deep_supervision, num_classes):
    model = UNet3Plus(pretrained=False, num_classes=num_classes, deep_supervision=deep_supervision,
                      num_side_classes=4).to(DEVICE)
    x = torch.randn(1, 3, SIZE, SIZE, device=DEVICE)
    model.train()
    with torch.no_grad():
        out = model(x)
    assert out["mask"].shape == (1, num_classes, SIZE, SIZE)
    assert out["centerline"].shape == (1, 1, SIZE, SIZE)
    if deep_supervision:
        assert len(out["side"]) == 4  # D2..D5
        assert all(s.shape == (1, 4, SIZE, SIZE) for s in out["side"])  # side heads: large classes only
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


def _targets(batch=2):
    """Strap-like bands, a thin cord, a large blob, a small hardware patch, and 1 px skeletons inside them."""
    m = torch.zeros(batch, len(CLASSES), SIZE, SIZE, device=DEVICE)
    m[:, 0, 300:310, :] = 1           # strap (horizontal band)
    m[:, 0, :, 400:408] = 1           # strap (vertical band)
    m[:, 1, 100:250, 100:600] = 1     # cover blob
    m[:, 2, 500:502, 50:700] = 1      # cord, 2 px
    m[:, 3, 302:308, 200:210] = 1     # hardware patch
    strap_skel = torch.zeros(batch, 1, SIZE, SIZE, device=DEVICE)
    strap_skel[:, :, 305, :] = 1
    strap_skel[:, :, :, 404] = 1
    cord_skel = torch.zeros_like(strap_skel)
    cord_skel[:, :, 500, 50:700] = 1
    cl3 = torch.zeros_like(strap_skel)
    cl3[:, :, 304:307, :] = 1
    return {"mask": m, "strap_cl_1px": strap_skel, "cord_cl_1px": cord_skel, "strap_cl_3px": cl3}


def _perfect_logits(t):
    return (t * 2 - 1) * 30.0  # sigmoid -> {~0, ~1}


@pytest.mark.parametrize("fn", [soft_dice_loss, focal_loss, soft_cldice_loss])
def test_mask_losses_finite_and_zero_when_perfect(fn):
    t = _targets()["mask"][:, :1]
    random_loss = fn(torch.randn_like(t), t)
    perfect_loss = fn(_perfect_logits(t), t)
    for v in (random_loss, perfect_loss):
        assert v.ndim == 0 and torch.isfinite(v)
    assert perfect_loss.item() < 1e-4
    assert random_loss.item() > perfect_loss.item()


def test_cldice_with_given_skeleton():
    t = _targets()
    mask, skel = t["mask"][:, :1], t["strap_cl_1px"]
    perfect = soft_cldice_loss(_perfect_logits(mask), mask, 5, skel_target=skel)
    broken = _perfect_logits(mask).clone()
    broken[..., :, 395:415] = -30.0  # cut the horizontal band: GT skeleton no longer covered
    assert perfect.item() < 1e-3 < soft_cldice_loss(broken, mask, 5, skel_target=skel).item()


@pytest.mark.parametrize("deep_supervision", [False, True])
def test_total_loss_finite_and_zero_when_perfect(deep_supervision):
    crit = SegLoss(LOSS_CFG, CLASSES, centerline_pos_weight=30.0,
                   side_classes=SIDE if deep_supervision else ()).to(DEVICE)
    t = _targets()
    side_idx = [CLASSES.index(c) for c in SIDE]
    perfect = {"mask": _perfect_logits(t["mask"]), "centerline": _perfect_logits(t["strap_cl_3px"])}
    noisy = {"mask": torch.randn_like(t["mask"]), "centerline": torch.randn_like(t["strap_cl_3px"])}
    if deep_supervision:
        perfect["side"] = [perfect["mask"][:, side_idx]] * 4
        noisy["side"] = [noisy["mask"][:, side_idx]] * 4
    total_p, terms_p = crit(perfect, t)
    total_n, terms_n = crit(noisy, t)
    assert torch.isfinite(total_p) and torch.isfinite(total_n) and total_p.ndim == 0
    assert all(math.isfinite(float(v)) for v in {**terms_p, **terms_n}.values())
    assert {f"dice/{c}" for c in CLASSES} <= set(terms_p) and {"cldice/strap", "cldice/cord_or_tie"} <= set(terms_p)
    assert terms_p["centerline_bce"] < 1e-4
    assert total_p.item() < 1e-2 < total_n.item()
    assert ("side" in terms_p) == deep_supervision


def test_cldice_weight_zero_is_skipped():
    cfg = {**LOSS_CFG, "cldice": {k: {**v, "weight": 0.0} for k, v in LOSS_CFG["cldice"].items()}}
    _, terms = SegLoss(cfg, CLASSES, 30.0).to(DEVICE)({"mask": torch.zeros(1, 4, 64, 64, device=DEVICE),
                                                       "centerline": torch.zeros(1, 1, 64, 64, device=DEVICE)},
                                                      {"mask": torch.zeros(1, 4, 64, 64, device=DEVICE),
                                                       "strap_cl_3px": torch.zeros(1, 1, 64, 64, device=DEVICE)})
    assert not any(k.startswith("cldice/") for k in terms)


def test_lr_schedule_freeze_warmup_cosine():
    spe, total = 10, 50  # 10 steps/epoch, 5 epochs
    dec = [lr_factor(s, 0, spe, total, 0.01) for s in range(total)]
    enc = [lr_factor(s, spe, spe, total, 0.01) for s in range(total)]
    assert all(v == 0.0 for v in enc[:spe])                      # encoder frozen in epoch 1
    assert enc[spe] == pytest.approx(0.1) and enc[2 * spe - 1] == pytest.approx(1.0)  # warmup in epoch 2
    assert dec[0] == pytest.approx(0.1) and dec[spe - 1] == pytest.approx(1.0)        # decoder warmup in epoch 1
    assert all(a >= b for a, b in zip(dec[spe:], dec[spe + 1:]))  # cosine: non-increasing
    assert dec[-1] == pytest.approx(0.01, abs=5e-3) and enc[-1] == pytest.approx(0.01, abs=5e-3)
