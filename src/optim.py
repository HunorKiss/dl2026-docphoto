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


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on the convolution and linear weights only, not on biases and norms (lab 4).

    Tensors with more than one dimension are convolution kernels and linear
    weight matrices; the one-dimensional ones are biases and the BatchNorm
    scales and shifts. Pulling those towards zero would only fight the
    normalisation, so they get no decay.
    """
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
