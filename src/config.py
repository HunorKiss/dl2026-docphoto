"""The config, and what every run sets up first: the seed, the device, the mixed precision."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "configs" / "reference.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device() -> torch.device:
    """The best available device: CUDA, then Apple MPS (the Mac GPU), then CPU.

    MPS is PyTorch's Metal backend; it runs only in a native-arch macOS
    process, never inside a Linux container, so this picks it up when the
    reference solution is run natively on an Apple Silicon Mac instead of
    through Docker.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(dev: torch.device, enabled: bool = True):
    """bfloat16 on a CUDA GPU that has it (Ampere and later); full precision everywhere else.

    Deliberately not enabled on MPS: bf16 autocast on the Metal backend is
    unreliable, and plain float32 on MPS is still far faster than emulated CPU.
    """
    on = enabled and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=on)
