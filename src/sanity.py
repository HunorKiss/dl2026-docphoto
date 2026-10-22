"""Two sanity checks to run before you trust a training curve.

Before a number means anything, two things have to hold. The model has to be
able to learn at all -- a bug in the loop, the loss or the model shows up as a
network that cannot even memorise a handful of rows. And a run has to be
repeatable -- if the same seed gives a different curve, you cannot tell an
improvement from noise, and every comparison this semester is worthless.

These are the first things to check when a run looks wrong, and the cheapest:
both run in seconds on a CPU.

    python -m src.sanity --config configs/lab3_mlp.yaml

`overfit_one_batch` drives a tiny batch to near-zero loss; if it cannot, the
model, the loss or the minibatch loop is broken, not the data. `same_seed_twice`
trains the same short run twice and checks the two agree: as src/seed.py warns,
two seeded runs agree closely but, on some GPU kernels, not bit for bit, so the
check is a tolerance, not an equality.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src import config
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.models.mlp import MLP
from src.seed import set_seed
from src.train import Standardiser, make_optimizer, predict, train_epoch, train_mlp


def overfit_one_batch(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray,
                      batch_size: int = 16, steps: int = 300, threshold: float = 1e-3,
                      device: str | None = None, log=print) -> dict:
    """Drive one small batch to near-zero loss, the "can this model learn at all" test.

    Take `batch_size` training rows, turn dropout off so there is nothing left
    to blame, and take `steps` gradient steps on that one batch. A model with a
    working loop and loss collapses to a tiny loss; a broken one plateaus. The
    data pipeline is deliberately not exercised -- the point is to isolate the
    model and the optimisation, so a failure here is never about the split or
    the labels.

    Returns the first and final loss and whether the final loss fell below
    `threshold`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    idx = train_idx[:batch_size]
    scaler = Standardiser().fit(X[idx])
    xb = torch.as_tensor(scaler.transform(X[idx]), dtype=torch.float32, device=device)
    yb = torch.as_tensor(df[config.TARGET].to_numpy()[idx], dtype=torch.float32, device=device)

    # Dropout off (eval mode keeps it off even inside the training step below),
    # so an inability to overfit cannot be blamed on regularisation.
    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=0.0,
                prior=float(yb.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()

    model.train()
    first_loss = None
    loss_val = float("nan")
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            log(f"step {step:4d}  loss {loss_val:.6f}")

    passed = loss_val < threshold
    log(f"overfit_one_batch: first {first_loss:.6f} -> final {loss_val:.6f}  "
        f"({'PASS' if passed else 'FAIL'}, threshold {threshold:g})")
    return {"first_loss": first_loss, "final_loss": loss_val, "passed": passed, "threshold": threshold}


def same_seed_twice(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray,
                    val_idx: np.ndarray, epochs: int = 3, atol: float = 1e-5,
                    device: str | None = None, log=print) -> dict:
    """Train the same short run twice with the same seed; the two must agree.

    Reproducibility is not optional: a comparison between two models only means
    something if each model gives the same curve when rerun. This runs the real
    training loop twice over `epochs` epochs with the config's seed untouched
    and compares the two validation-RMSE histories.

    As src/seed.py notes, seeded runs agree closely but, on some GPU kernels,
    not bit for bit, so the test is `np.allclose` within `atol`, not equality.
    W&B logging is forced to disabled for the duration: the check is about the
    model, not the tracker, and two throwaway runs should not litter the board.
    The runs use a shortened copy of the config (`epochs` epochs) to stay fast.

    Returns the two histories, their largest per-epoch difference and the verdict.
    """
    short = dict(cfg)
    short["train"] = dict(cfg["train"], epochs=epochs)

    prev_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"          # the check is about the model, not the tracker
    try:
        first = train_mlp(short, X, df, train_idx, val_idx, device=device, log=lambda *a, **k: None)
        second = train_mlp(short, X, df, train_idx, val_idx, device=device, log=lambda *a, **k: None)
    finally:
        if prev_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = prev_mode

    v1 = first["history"]["val_rmse"].to_numpy()
    v2 = second["history"]["val_rmse"].to_numpy()
    max_diff = float(np.max(np.abs(v1 - v2))) if len(v1) == len(v2) else float("inf")
    passed = bool(len(v1) == len(v2) and np.allclose(v1, v2, atol=atol))

    for e, (a, b) in enumerate(zip(v1, v2), start=1):
        log(f"epoch {e:3d}  run A val RMSE {a:.6f}  run B {b:.6f}  |diff| {abs(a - b):.2e}")
    log(f"same_seed_twice: max |diff| {max_diff:.2e}  "
        f"({'PASS' if passed else 'FAIL'}, atol {atol:g})")
    return {"val_rmse_a": v1, "val_rmse_b": v2, "max_diff": max_diff, "passed": passed, "atol": atol}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the training sanity checks from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"],
                                      cfg["split"]["val_fraction"], cfg["split"]["seed"])

    print("== overfit_one_batch ==")
    r1 = overfit_one_batch(cfg, X, df, train_idx)
    print("\n== same_seed_twice ==")
    r2 = same_seed_twice(cfg, X, df, train_idx, val_idx)

    if not (r1["passed"] and r2["passed"]):
        raise SystemExit("a sanity check failed; fix it before trusting a training curve")


if __name__ == "__main__":
    main()
