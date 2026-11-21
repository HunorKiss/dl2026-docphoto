"""Predictions for a loader of photographs, and the metric they are scored with."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import autocast, device
from src.loss import to_prediction


@torch.no_grad()
def predict(model: nn.Module, loader, amp: bool = True, tta: bool = False) -> np.ndarray:
    """Usability in [0, 1] for every photograph of the loader, in order.

    With `tta`, each photograph is predicted twice, as it is and mirrored
    left-right, and the two predictions are averaged. Mirroring is one of the
    two augmentations training uses, so it is a view the network knows, and it
    changes nothing about image quality. Each photograph is still predicted on
    its own: nothing is shared between photographs.
    """
    dev = device()
    model.to(dev).eval()
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=dev.type == "cuda")     # asynchronous copies only where they are safe
        with autocast(dev, amp):
            p = to_prediction(model(x))
            if tta:
                p = (p + to_prediction(model(x.flip(-1)))) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
