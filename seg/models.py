"""U-Net style segmenters on ImageNet encoders (VGG-16, VGG-19, InceptionV3),
with optional additive attention gates on the skip connections (Oktay et al. 2018)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from . import ROOT  # noqa: F401  (sets TORCH_HOME)


class VGGEncoder(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        # batch-norm variants of VGG-16/19 train far more stably with a fresh decoder
        ctor = {"vgg16": tvm.vgg16_bn, "vgg19": tvm.vgg19_bn}[name]
        weights = "DEFAULT" if pretrained else None
        feats = ctor(weights=weights).features
        # tap the ReLU before each max-pool: strides 1, 2, 4, 8, 16
        pools = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]
        taps = [p - 1 for p in pools][:5]
        self.stages = nn.ModuleList()
        start = 0
        for t in taps:
            self.stages.append(feats[start:t + 1])
            start = t + 1
        self.channels = [64, 128, 256, 512, 512]

    def forward(self, x):
        out = []
        for s in self.stages:
            x = s(x)
            out.append(x)
        return out


class InceptionEncoder(nn.Module):
    """torchvision InceptionV3 cut into five stages at strides 2, 4, 8, 16, 32.
    Its valid-padded stems give odd sizes, so the decoder resizes to each skip."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        m = tvm.inception_v3(weights=weights, aux_logits=True, init_weights=not pretrained)
        self.stages = nn.ModuleList([
            nn.Sequential(m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3),
            nn.Sequential(m.maxpool1, m.Conv2d_3b_1x1, m.Conv2d_4a_3x3),
            nn.Sequential(m.maxpool2, m.Mixed_5b, m.Mixed_5c, m.Mixed_5d),
            nn.Sequential(m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e),
            nn.Sequential(m.Mixed_7a, m.Mixed_7b, m.Mixed_7c),
        ])
        self.channels = [64, 192, 288, 768, 2048]

    def forward(self, x):
        out = []
        for s in self.stages:
            x = s(x)
            out.append(x)
        return out


class AttentionGate(nn.Module):
    """Additive attention: alpha = sigmoid(psi(relu(Wx x + Wg g))), skip * alpha."""

    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.wx = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.wg = nn.Conv2d(gate_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.last_alpha = None
        self.ablate = None  # test-time only: "mean" keeps each map's per-slice mean, "ones" opens the gate

    def forward(self, x, g):
        g = F.interpolate(self.wg(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.wx(x) + g)))
        self.last_alpha = alpha.detach()
        if self.ablate == "mean":
            alpha = alpha.mean((-2, -1), keepdim=True)
        elif self.ablate == "ones":
            alpha = torch.ones_like(alpha)
        return x * alpha


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, attention):
        super().__init__()
        self.gate = AttentionGate(skip_ch, in_ch, max(skip_ch // 2, 16)) if attention else None
        self.conv = conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        if self.gate is not None:
            skip = self.gate(skip, x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SegNet(nn.Module):
    def __init__(self, encoder="vgg16", attention=False, pretrained=True, dec_ch=(256, 128, 64, 32), prior=0.02,
                 in_scale=1):
        super().__init__()
        self.in_scale = in_scale  # upsample the input this much inside the net; output keeps input size
        if encoder.startswith("vgg"):
            self.encoder = VGGEncoder(encoder, pretrained)
        elif encoder == "inceptionv3":
            self.encoder = InceptionEncoder(pretrained)
        else:
            raise ValueError(encoder)
        ch = self.encoder.channels
        self.blocks = nn.ModuleList()
        prev = ch[-1]
        for skip_ch, out_ch in zip(reversed(ch[:-1]), dec_ch):
            self.blocks.append(DecoderBlock(prev, skip_ch, out_ch, attention))
            prev = out_ch
        self.head = nn.Conv2d(prev, 1, 1)
        # start at the tumour prior (~2% of pixels) instead of p = 0.5; with a zero bias,
        # background that the decoder maps to near-zero features is stuck near p = 0.3
        # for hundreds of steps (Lin et al. 2017, focal loss prior init)
        nn.init.constant_(self.head.bias, math.log(prior / (1 - prior)))
        self.encoder_name, self.attention = encoder, attention

    def forward(self, x):
        size = x.shape[-2:]
        if self.in_scale != 1:
            x = F.interpolate(x, scale_factor=self.in_scale, mode="bilinear", align_corners=False)
        feats = self.encoder(x)
        y = feats[-1]
        for blk, skip in zip(self.blocks, reversed(feats[:-1])):
            y = blk(y, skip)
        return F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)

    def attention_maps(self):
        """Gate coefficients from the last forward pass, finest first."""
        return [b.gate.last_alpha for b in reversed(self.blocks) if b.gate is not None]
