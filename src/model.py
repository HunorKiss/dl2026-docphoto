"""The network: lab 4's SmallCNN, how the config builds it, and how a checkpoint gives it back.

The network returns a logit. Turning that into a usability prediction is the
job of src/loss.py, so that the loss and the prediction can never disagree.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int, stride: int = 1, kernel: int = 3) -> nn.Sequential:
    """Convolution, BatchNorm, ReLU. No bias in the convolution: the norm would subtract it again."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Lab 4's network: a strided stem, conv-pool blocks, average and maximum pooling, one logit.

    The 7 x 7 stride-4 stem reads every pixel of the 768 px page and hands a
    192 px map to the blocks; that is what made 768 px affordable in lab 4.
    The pooling concatenates the mean over all positions ("how sharp is the
    page overall") with the maximum ("how strong is the strongest response
    anywhere"). The forward pass returns a logit; src/loss.py turns it into a
    prediction.

    The network standardises its own input with the mean and standard
    deviation of the training pixels, stored in the checkpoint, so that
    prediction cannot forget to do it.
    """

    def __init__(self, widths=(8, 16, 32, 64), stem_stride: int = 4, stem_kernel: int = 7,
                 dropout: float = 0.0, prior: float = 0.5):
        super().__init__()
        layers = [conv_block(1, widths[0], stride=stem_stride, kernel=stem_kernel)]
        for c_in, c_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [conv_block(c_in, c_out), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * widths[-1], 1)
        self.register_buffer("in_mean", torch.tensor(0.5))
        self.register_buffer("in_std", torch.tensor(0.25))

        # He initialisation for the ReLU stack (lecture 4); the head starts
        # with tiny weights and its bias at logit(prior), so that step 0
        # predicts the training mean instead of 0.5.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features((x - self.in_mean) / self.in_std)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        return self.head(self.drop(f)).squeeze(1)


class MaxHalfScale(nn.Module):
    """Multiplies the second half of the pooled features (the maxima) by one learnable number.

    It starts at `init`, the ratio of the feature map's mean to its maximum on the
    training photographs, so that the means and the maxima reach the head on the
    same scale. One number for the whole half: what differs between channels and
    between photographs is left as it is.
    """

    def __init__(self, init: float = 0.02):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        half = f.shape[1] // 2
        return torch.cat([f[:, :half], f[:, half:] * self.scale], dim=1)


class EfficientNetB0Reg(nn.Module):
    """An ImageNet-pretrained EfficientNet-B0 with a one-logit regression head.

    The rest of the pipeline feeds a 1-channel grey image in [0, 1] (src/data.py);
    a pretrained backbone expects 3 channels normalised with ImageNet statistics.
    Like SmallCNN standardises its own input, this model adapts its own input so
    data.py and the training loop stay unchanged: grey is repeated to 3 channels
    and ImageNet-normalised inside forward(). The head starts at logit(prior) so
    step 0 predicts the training mean. `set_backbone_frozen` freezes every weight
    except the head, for the "head" training phases (src/train.py).

    `pool` decides what the head sees of the backbone's 1280 x 24 x 24 feature map:
      "avg"     the mean over all positions (torchvision's own head): 1280 numbers.
      "avgmax"  the mean and the maximum side by side, as in SmallCNN: 2560 numbers,
                the max half scaled by a learnable number (`pool_norm`) before the head.
                The mean says how sharp the page is overall, the maximum how strong
                the strongest response anywhere is, so a defect in one region of the
                page is not averaged away over the other positions.
    """

    POOLS = ("avg", "avgmax")

    def __init__(self, dropout: float = 0.2, prior: float = 0.5, pretrained: bool = True, pool: str = "avg"):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        if pool not in self.POOLS:
            raise ValueError(f"unknown pool {pool!r}; use one of {self.POOLS}")
        self.pool = pool
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features * (2 if pool == "avgmax" else 1)
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        self.head = self.backbone.classifier[1]          # so sanity.py / optim.py can find `head`
        # The maxima of the feature map are about 50 times larger than its means
        # (measured on the training photographs: 8.1 against 0.165 on average).
        # Fed side by side to one linear layer, the max half dominates: AdamW moves
        # every weight by about the same step, so the logit jumps by tens per step,
        # the sigmoid saturates and learning stops. `MaxHalfScale` multiplies the
        # max half by a learnable number that starts at that ratio, 1/50, so both
        # halves enter the head on the same scale. It uses no batch statistics and
        # changes nothing between photographs. (A BatchNorm here kept oscillating on
        # small batches.) It belongs to the head, so it trains also while the
        # backbone is frozen. With "avg" it is the identity: those models are
        # exactly as before, and so are their checkpoints.
        self.pool_norm = MaxHalfScale() if pool == "avgmax" else nn.Identity()

        # ImageNet input statistics, kept as buffers so they travel with the checkpoint.
        m = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        self.register_buffer("norm_mean", torch.tensor(m.mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(m.std).view(1, 3, 1, 1))
        # Kept only for checkpoint compatibility with the SmallCNN interface; unused here.
        self.register_buffer("in_mean", torch.tensor(0.5))
        self.register_buffer("in_std", torch.tensor(0.25))

        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))

    def head_parameters(self) -> list[nn.Parameter]:
        """The regression head (and, with pool "avgmax", its max-half scale), for train.finetune_head_lr."""
        return list(self.backbone.classifier.parameters()) + list(self.pool_norm.parameters())

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze every backbone weight; the head always stays trainable."""
        self._backbone_frozen = frozen
        for name, p in self.backbone.named_parameters():
            p.requires_grad = not frozen
        for p in self.head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the backbone's BatchNorm layers in eval mode while the backbone is frozen.

        A frozen backbone must not update its BatchNorm running statistics: with
        a small batch they drift over many steps, the backbone's output blows up,
        and the loss becomes inf. Only the head (and the backbone once unfrozen)
        follows the requested train/eval mode.
        """
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            self.backbone.eval()
            self.head.train(mode)      # the head still trains
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)                         # 1 grey channel -> 3
        x = (x - self.norm_mean) / self.norm_std         # ImageNet normalisation
        f = self.backbone.features(x)                    # (batch, 1280, h, w)
        if self.pool == "avg":
            f = torch.flatten(self.backbone.avgpool(f), 1)   # exactly torchvision's EfficientNet.forward
        else:
            f = self.pool_norm(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return self.backbone.classifier(f).squeeze(1)    # logit; the sigmoid lives in src/loss.py


class ConvNeXtTinyReg(nn.Module):
    """An ImageNet-pretrained ConvNeXt-Tiny with a small MLP regression head: 768 -> hidden -> 1.

    ConvNeXt is a modernised CNN: a 4 x 4 stride-4 stem, then four stages with
    7 x 7 depthwise convolutions, LayerNorm and an inverted-bottleneck MLP, each
    stage halving the resolution. A 768 px page becomes a 768 x 24 x 24 feature
    map; the network is fully convolutional, so 768 px needs no change to it.
    It normalises with LayerNorm, not BatchNorm, so it does not depend on the
    batch size, which at 768 px has to stay small.

    As in EfficientNetB0Reg, the grey 1-channel input is repeated to 3 channels
    and ImageNet-normalised inside forward(), so data.py and the training loop
    stay unchanged. The pooled features go through torchvision's own pretrained
    LayerNorm, then the head: Linear(768, hidden), GELU, Dropout, Linear(hidden, 1).
    The head returns a logit; the sigmoid is applied in src/loss.py.

    The head (with that LayerNorm) lives outside `self.backbone`, so freezing
    the backbone never switches the head's dropout off. `head_parameters()`
    lets train.py give the head its own learning rate (train.finetune_head_lr).
    """

    def __init__(self, dropout: float = 0.2, prior: float = 0.5, pretrained: bool = True, hidden: int = 256):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = net.features                       # stem + 4 stages: (batch, 768, h/32, w/32)
        self.norm = net.classifier[0]                      # LayerNorm2d(768), pretrained with the backbone
        width = net.classifier[2].in_features              # 768
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        m = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        self.register_buffer("norm_mean", torch.tensor(m.mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(m.std).view(1, 3, 1, 1))
        # Kept only for checkpoint compatibility with the SmallCNN interface; unused here.
        self.register_buffer("in_mean", torch.tensor(0.5))
        self.register_buffer("in_std", torch.tensor(0.25))

        # The last layer starts with tiny weights and its bias at logit(prior):
        # step 0 predicts the training mean, as in the other models.
        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.constant_(self.head[-1].bias, math.log(p / (1 - p)))

    def head_parameters(self) -> list[nn.Parameter]:
        """Everything after the backbone: the pretrained LayerNorm and the MLP."""
        return list(self.norm.parameters()) + list(self.head.parameters())

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze every backbone weight; the head always stays trainable."""
        self._backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        for p in self.head_parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """While the backbone is frozen it stays in eval mode (no stochastic depth); the head follows `mode`."""
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)                           # 1 grey channel -> 3
        x = (x - self.norm_mean) / self.norm_std           # ImageNet normalisation
        f = self.backbone(x).mean((2, 3), keepdim=True)    # global average pool, as torchvision's avgpool
        f = self.norm(f).flatten(1)                        # (batch, 768)
        return self.head(f).squeeze(1)                     # logit; the sigmoid lives in src/loss.py


def build_model(cfg: dict, prior: float = 0.5):
    m = cfg["model"]
    kind = m.get("type", "small_cnn")
    if kind == "small_cnn":
        return SmallCNN(m["widths"], m["stem_stride"], m["stem_kernel"], m["dropout"], prior)
    if kind == "effnet_b0":
        # model.pool defaults to "avg", so configs and checkpoints from before it existed are unchanged
        return EfficientNetB0Reg(m.get("dropout", 0.2), prior, m.get("pretrained", True), m.get("pool", "avg"))
    if kind == "convnext_tiny":
        return ConvNeXtTinyReg(m.get("dropout", 0.2), prior, m.get("pretrained", True), m.get("hidden", 256))
    raise ValueError(f"unknown model type {kind!r}")


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    # The checkpoint holds every weight, so the ImageNet weights need not be
    # downloaded again just to be overwritten.
    build_cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(build_cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg
