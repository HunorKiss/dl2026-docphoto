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


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(dev: torch.device, enabled: bool = True):
    """bfloat16 on a GPU that has it (Ampere and later); full precision everywhere else."""
    on = enabled and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=on)
