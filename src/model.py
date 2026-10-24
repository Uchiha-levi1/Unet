"""UNet 3+ (Huang et al., 2020) with a timm ResNet encoder, without the classification-guided module."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FullScaleDecoderLevel(nn.Module):
    """Decoder level D_i fed by full-scale skip connections.

    Levels are numbered 1 (finest, stride 2) to 5 (coarsest, stride 32). D_i receives:
      * finer encoder maps E_1..E_{i-1}, max-pooled down to level i's resolution,
      * the same-scale encoder map E_i,
      * deeper decoder maps D_{i+1}..D_5, bilinearly upsampled to level i (D_5 is E_5).
    That is always 5 inputs. Each is projected to `branch` channels (3x3 conv-BN-ReLU),
    concatenated to 5*branch channels, and fused with one more 3x3 conv-BN-ReLU.
    """

    def __init__(self, level, enc_channels, dec_channels, branch):
        super().__init__()
        self.level = level
        self.enc_branches = nn.ModuleList(conv_bn_relu(enc_channels[j], branch) for j in range(level))
        self.dec_branches = nn.ModuleList(conv_bn_relu(dec_channels[j], branch) for j in range(level, 5))
        self.fuse = conv_bn_relu(5 * branch, 5 * branch)

    def forward(self, enc, dec):
        # enc: [E1..E5]; dec: {level: D_level} for every level deeper than self.level.
        i = self.level - 1  # 0-based index of this level
        size = enc[i].shape[-2:]
        feats = []
        for j, branch in enumerate(self.enc_branches):
            x = enc[j]
            if j < i:  # finer level: stride doubles per level, so pool by 2^(i-j)
                k = 2 ** (i - j)
                x = F.max_pool2d(x, kernel_size=k, stride=k, ceil_mode=True)
            feats.append(branch(x))
        for j, branch in zip(range(i + 1, 5), self.dec_branches):
            x = F.interpolate(dec[j + 1], size=size, mode="bilinear", align_corners=False)
            feats.append(branch(x))
        return self.fuse(torch.cat(feats, dim=1))


class UNet3Plus(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, num_classes=1, branch_channels=64,
                 deep_supervision=False, grad_checkpointing=False):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        enc_ch = self.encoder.feature_info.channels()
        assert len(enc_ch) == 5 and self.encoder.feature_info.reduction() == [2, 4, 8, 16, 32]
        fused = 5 * branch_channels
        dec_ch = [fused] * 4 + [enc_ch[4]]  # D1..D4 have `fused` channels; D5 = E5
        self.decoder = nn.ModuleDict({str(l): FullScaleDecoderLevel(l, enc_ch, dec_ch, branch_channels)
                                      for l in (4, 3, 2, 1)})

        # Heads on D1 (stride 2): 3x3 conv to logits, then bilinear upsample to input size.
        self.mask_head = nn.Conv2d(fused, num_classes, 3, padding=1)
        self.centerline_head = nn.Conv2d(fused, 1, 3, padding=1)

        self.deep_supervision = deep_supervision
        if deep_supervision:  # side mask heads on D2..D5
            self.side_heads = nn.ModuleDict({str(l): nn.Conv2d(dec_ch[l - 1], num_classes, 3, padding=1)
                                             for l in (2, 3, 4, 5)})

        # Checkpointing covers the decoder levels only: they hold most of the activation memory
        # (320-channel maps up to stride 2). timm's encoder checkpointing re-runs in-place ReLUs
        # during backward, which corrupts E1 as saved by the decoder checkpoints.
        self.grad_checkpointing = grad_checkpointing
        self.freeze_encoder_bn = False

    def train(self, mode=True):
        super().train(mode)
        # With small batches, keep the pretrained encoder BN running stats frozen.
        if mode and self.freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        enc = self.encoder(x)
        dec = {5: enc[4]}
        for level in (4, 3, 2, 1):
            block = self.decoder[str(level)]
            if self.grad_checkpointing and self.training:
                dec[level] = checkpoint(block, enc, dec, use_reentrant=False)
            else:
                dec[level] = block(enc, dec)

        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        out = {"mask": up(self.mask_head(dec[1])), "centerline": up(self.centerline_head(dec[1]))}
        if self.deep_supervision and self.training:
            out["side"] = [up(self.side_heads[str(l)](dec[l])) for l in (2, 3, 4, 5)]
        return out


def build_model(model_cfg):
    return UNet3Plus(encoder=model_cfg["encoder"], pretrained=model_cfg["pretrained"],
                     num_classes=len(model_cfg["classes"]), branch_channels=model_cfg["branch_channels"],
                     deep_supervision=model_cfg["deep_supervision"],
                     grad_checkpointing=model_cfg["grad_checkpointing"])
