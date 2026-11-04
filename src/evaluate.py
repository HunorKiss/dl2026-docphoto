"""Predictions for a loader of photographs, and the metric they are scored with."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import autocast, device
from src.loss import to_prediction


@torch.no_grad()
def predict(model: nn.Module, loader, amp: bool = True) -> np.ndarray:
    """Usability in [0, 1] for every photograph of the loader, in order."""
    dev = device()
    model.to(dev).eval()
    out = []
    for x, _ in loader:
        with autocast(dev, amp):
            out.append(to_prediction(model(x.to(dev, non_blocking=True))).cpu())
    return torch.cat(out).numpy()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
