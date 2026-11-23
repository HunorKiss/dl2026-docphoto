"""The optimiser's parameter groups and the learning-rate schedule (lab 4)."""
from __future__ import annotations

import math

import torch


def lr_at(step: int, total: int, warmup: float) -> float:
    """Linear warm-up, then a cosine down to zero; a factor on the base learning rate.

    `step` counts optimiser steps (batches), not epochs. The first `warmup`
    fraction of them ramps the factor from almost 0 to 1, so that the first,
    still random updates cannot throw the network far off; the rest follows
    half a cosine from 1 to 0, so that the last epoch makes only small changes
    and the final weights are a settled point rather than a random one.
    """
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def _no_decay(p: torch.Tensor) -> bool:
    """One number per channel: biases, norm scales and shifts, ConvNeXt's layer scale.

    ConvNeXt (torchvision) stores its layer scale as a (C, 1, 1) tensor; it is a
    per-channel gain like a norm's, so it gets no decay either. Every
    convolution kernel and linear weight matrix has a second dimension larger
    than one and keeps its decay.
    """
    return p.ndim <= 1 or all(s == 1 for s in p.shape[1:])


def param_groups(model: torch.nn.Module, weight_decay: float, head_params=None,
                 head_lr: float | None = None) -> list[dict]:
    """Weight decay on the convolution and linear weights only, not on biases and norms (lab 4).

    Tensors with more than one dimension are convolution kernels and linear
    weight matrices; the one-dimensional ones are biases and the BatchNorm
    scales and shifts. Pulling those towards zero would only fight the
    normalisation, so they get no decay.

    With `head_params` and `head_lr`, those parameters (the regression head) get
    their own learning rate, `head_lr`; every other parameter uses the
    optimiser's lr. The weight decay rule is the same for both.
    """
    head_ids = {id(p) for p in head_params} if head_params is not None else set()
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            ps = [p for p in model.parameters() if (id(p) in head_ids) == is_head and _no_decay(p) != decay]
            if ps:
                g = {"params": ps, "weight_decay": weight_decay if decay else 0.0}
                if is_head:
                    g["lr"] = head_lr
                groups.append(g)
    return groups
