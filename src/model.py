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


class EfficientNetB0Reg(nn.Module):
    """An ImageNet-pretrained EfficientNet-B0 with a one-logit regression head.

    The rest of the pipeline feeds a 1-channel grey image in [0, 1] (src/data.py);
    a pretrained backbone expects 3 channels normalised with ImageNet statistics.
    Like SmallCNN standardises its own input, this model adapts its own input so
    data.py and the training loop stay unchanged: grey is repeated to 3 channels
    and ImageNet-normalised inside forward(). The head starts at logit(prior) so
    step 0 predicts the training mean. `freeze_backbone` freezes every weight
    except the head, for a first phase of training the head alone; train.py
    unfreezes it after `train.freeze_epochs` epochs.
    """

    def __init__(self, dropout: float = 0.2, prior: float = 0.5, pretrained: bool = True):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        self.head = self.backbone.classifier[1]          # so sanity.py / optim.py can find `head`

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
        return self.backbone(x).squeeze(1)               # logit; the sigmoid lives in src/loss.py


def build_model(cfg: dict, prior: float = 0.5):
    m = cfg["model"]
    kind = m.get("type", "small_cnn")
    if kind == "small_cnn":
        return SmallCNN(m["widths"], m["stem_stride"], m["stem_kernel"], m["dropout"], prior)
    if kind == "effnet_b0":
        return EfficientNetB0Reg(m.get("dropout", 0.2), prior, m.get("pretrained", True))
    raise ValueError(f"unknown model type {kind!r}")


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg
