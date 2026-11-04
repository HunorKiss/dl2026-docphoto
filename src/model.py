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


def build_model(cfg: dict, prior: float = 0.5) -> SmallCNN:
    m = cfg["model"]
    return SmallCNN(m["widths"], m["stem_stride"], m["stem_kernel"], m["dropout"], prior)


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg
