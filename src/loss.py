"""The loss, and how the network's output becomes a prediction: one pair, kept in one file.

The network returns a raw number (a logit). `to_prediction` maps it into
[0, 1], the range of the label; `loss_fn` scores that prediction. Training
(train.py), the checks (scripts/sanity.py) and prediction (src/evaluate.py,
hence predict.py) all use these two, so if you change what the output means
(another activation, classes instead of a number, ...), change both here and
every part of the pipeline follows. Changing one without the other trains a
model whose predictions mean something else than what it was trained for,
and nothing raises an error.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def to_prediction(out: torch.Tensor) -> torch.Tensor:
    """The network's raw output as usability in [0, 1]: the sigmoid, in full precision."""
    return torch.sigmoid(out.float())


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE of the prediction against the label: the loss is the square of the metric (RMSE)."""
    return nn.functional.mse_loss(to_prediction(out), y)
