#!/usr/bin/env python3
"""Confirm the container is set up correctly before you debug anything else.

    docker compose run --rm dev python scripts/check_env.py
"""
from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python       {sys.version.split()[0]} on {platform.system()}")

    import numpy
    import pandas
    import sklearn
    print(f"numpy        {numpy.__version__}")
    print(f"pandas       {pandas.__version__}")
    print(f"scikit-learn {sklearn.__version__}")

    import cv2
    print(f"opencv       {cv2.__version__}")

    import torch
    import torchvision
    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")

    if torch.cuda.is_available():
        print(f"gpu          {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        print(f"gpu check    matmul ok, result {float((x @ x).sum()):.1f}")
    else:
        print("gpu          not available; training will run on CPU and be slow")

    import os

    import wandb
    key = "set" if os.environ.get("WANDB_API_KEY") else "not set, runs will be anonymous"
    print(f"wandb        {wandb.__version__}, mode {os.environ.get('WANDB_MODE', 'online')}, api key {key}")

    from src.config import ROOT
    print(f"project root {ROOT}")
    raw = ROOT / "data" / "raw"
    for name, want in (("train.csv", None), ("train", 1841), ("test", 1628),
                       ("sample_submission.csv", None)):
        p = raw / name
        if want is None:
            state = "ok" if p.is_file() else "MISSING"
        else:
            n = len(list(p.glob("*.jpg"))) if p.is_dir() else 0
            state = f"{n} .jpg" + ("" if n == want else f", expected {want}")
        print(f"data/raw     {name}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
