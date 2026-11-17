"""Data_v3 options: two centerline channels, ignore region (data_tag), consistency loss, head remapping init."""
import torch

from src.losses import SegLoss
from src.model import UNet3Plus
from src.train import load_init_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S = 128
CLASSES = ["webbing", "cargo_cover", "cord_or_tie", "attachment_hardware", "tape"]
LOSS_CFG = {"class_weights": {"webbing": 1.0, "cargo_cover": 0.25, "cord_or_tie": 1.5, "attachment_hardware": 1.0,
                              "tape": 0.5, "strap": 1.0},
            "focal_gamma": 2.0, "focal_alpha": None, "centerline_bce": 1.0,
            "cldice": {"webbing": {"weight": 2.0, "iters": 5, "skeleton": "webbing_cl_1px"},
                       "cord_or_tie": {"weight": 2.0, "iters": 3, "skeleton": "cord_cl_1px"},
                       "strap": {"weight": 2.0, "iters": 5, "skeleton": "strap_cl_1px"}},  # not a class: skipped
            "consistency": {"weight": 0.5, "support": ["webbing", "cord_or_tie", "tape", "attachment_hardware"]}}
CL = ["webbing_cl_3px", "cord_cl_3px"]


def _targets():
    m = torch.zeros(2, len(CLASSES), S, S, device=DEVICE)
    m[:, 0, 50:60, :] = 1
    m[:, 1, 0:40, :] = 1
    m[:, 2, 90:92, :] = 1
    t = {"mask": m}
    for k, rows in (("webbing_cl_1px", [55]), ("webbing_cl_3px", [54, 55, 56]),
                    ("cord_cl_1px", [90]), ("cord_cl_3px", [90, 91])):
        x = torch.zeros(2, 1, S, S, device=DEVICE)
        x[:, :, rows, :] = 1
        t[k] = x
    t["ignore"] = torch.zeros(2, 1, S, S, device=DEVICE)
    t["ignore"][:, :, 100:120, 10:30] = 1
    return t


def _perfect(x):
    return (x * 2 - 1) * 30.0


def _crit():
    return SegLoss(LOSS_CFG, CLASSES, [29.6, 30.0], centerline_targets=CL).to(DEVICE)


def test_two_centerline_heads():
    out = UNet3Plus(pretrained=False, num_classes=8, num_centerlines=2).eval()(torch.randn(1, 3, 256, 256))
    assert out["mask"].shape == (1, 8, 256, 256) and out["centerline"].shape == (1, 2, 256, 256)


def test_v3_loss_perfect_vs_noisy():
    t = _targets()
    cl = torch.cat([t[k] for k in CL], 1)
    total_p, terms = _crit()({"mask": _perfect(t["mask"]), "centerline": _perfect(cl)}, t)
    total_n, _ = _crit()({"mask": torch.randn_like(t["mask"]), "centerline": torch.randn_like(cl)}, t)
    assert {"consistency/webbing_cl_3px", "consistency/cord_cl_3px", "cldice/webbing"} <= set(terms)
    assert "cldice/strap" not in terms
    assert total_p.item() < 1e-2 < total_n.item()


def test_ignore_region_has_no_effect():
    t = _targets()
    cl = torch.cat([t[k] for k in CL], 1)
    a = {"mask": _perfect(t["mask"]), "centerline": _perfect(cl)}
    b = {k: v.clone() for k, v in a.items()}
    b["mask"][:, :, 100:120, 10:30] = 30.0       # garbage inside the ignore region only
    b["centerline"][:, :, 100:120, 10:30] = 30.0
    la, lb = _crit()(a, t)[0], _crit()(b, t)[0]
    assert torch.allclose(la, lb, atol=1e-5)


def test_consistency_penalizes_centerline_off_support():
    t = _targets()
    cl = torch.cat([t[k] for k in CL], 1)
    good = {"mask": _perfect(t["mask"]), "centerline": _perfect(cl)}
    bad = {"mask": good["mask"], "centerline": good["centerline"].clone()}
    bad["centerline"][:, 0, 70:75, :] = 30.0      # webbing centerline where no support class is predicted
    tg, tb = _crit()(good, t)[1], _crit()(bad, t)[1]
    assert tb["consistency/webbing_cl_3px"] > 0.3 > tg["consistency/webbing_cl_3px"]


def test_init_head_map(tmp_path):
    old_classes = ["strap", "cargo_cover", "cord_or_tie", "hardware", "tape", "skid_board", "honeycomb",
                   "parachute_bag", "data_tag", "pack_band"]
    src = UNet3Plus(pretrained=False, num_classes=10)
    torch.save({"model": src.state_dict(), "epoch": 12}, tmp_path / "best.pt")
    dst = UNet3Plus(pretrained=False, num_classes=5, num_centerlines=2)
    fresh_cord_cl = dst.centerline_head.weight[1].clone()
    head_map = {"classes": old_classes, "mask": {"webbing": "strap", "attachment_hardware": "hardware",
                                                 "tape": "tape"}, "centerline": {0: 0}}
    assert load_init_weights(dst, tmp_path / "best.pt", head_map, CLASSES) == 12
    s, d = src.state_dict(), dst.state_dict()
    assert torch.equal(d["mask_head.weight"][0], s["mask_head.weight"][0])   # webbing <- strap
    assert torch.equal(d["mask_head.weight"][3], s["mask_head.weight"][3])   # attachment_hardware <- hardware
    assert torch.equal(d["mask_head.bias"][4], s["mask_head.bias"][4])       # tape <- tape
    assert torch.equal(d["centerline_head.weight"][0], s["centerline_head.weight"][0])
    assert torch.equal(d["centerline_head.weight"][1], fresh_cord_cl)        # cord centerline: fresh
    assert torch.equal(d["decoder.1.fuse.0.weight"], s["decoder.1.fuse.0.weight"])
