"""The checks before training (lab 4): run them whenever you change the network, the loss or the input.

    python scripts/sanity.py                                   # the reference config
    python scripts/sanity.py --config configs/my_try.yaml      # your change
    python scripts/sanity.py --config configs/my_try.yaml && python -m src.train --fold 0 --config ...

A run that "does not learn" looks the same whatever the cause. Each check takes
seconds (a minute or two on a CPU) and rules out a family of causes:

  1. one forward pass: one finite number per photograph, and a prediction in [0, 1];
  2. the loss before training: the untrained network predicts the training
     mean (its output bias starts there), so its loss is about the loss of
     that constant. Far above it, the initialisation or the input
     standardisation is wrong;
  3. overfit one batch: trained on 16 photographs and nothing else, the loss
     must fall to nearly zero. A network that cannot do that has a bug (a
     wrong target, a detached graph, frozen weights, a learning rate of zero),
     and more data will not fix it;
  4. the same seed twice: two short runs from the same seed give the same
     losses, or no comparison between two of your runs means anything.

It uses the network (src/model.py), the loss and the prediction (src/loss.py)
and the image size of the config, on photographs from the resize cache (run
src.prepare first). It
exits with 1 if a check fails, so it can stop a long job before it starts.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from src.config import autocast, device, load_config, seed_everything
from src.data import TARGET, Photos, cache_path, train_table
from src.loss import loss_fn, to_prediction
from src.model import build_model


def one_batch(cfg: dict, n: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """n training photographs spread over the table, without augmentation, and the training mean."""
    df = train_table(cfg)
    rows = df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]
    paths = [cache_path(cfg, "train", i) for i in rows["image_id"]]
    if not paths[0].exists():
        sys.exit(f"{paths[0]} is missing: run python -m src.prepare first")
    ds = Photos(paths, rows[TARGET], size=cfg["data"]["image_size"])
    x, y = zip(*(ds[i] for i in range(n)), strict=True)
    return torch.stack(x), torch.stack(y), float(df[TARGET].mean())


def fresh(cfg: dict, prior: float, x: torch.Tensor, dev: torch.device) -> torch.nn.Module:
    """A new network as train.py builds it: output bias at the training mean, input standardised."""
    model = build_model(cfg, prior=prior)
    model.in_mean.fill_(float(x.mean()))
    model.in_std.fill_(float(x.std()))
    return model.to(dev)


def train_steps(model, x, y, steps: int, lr: float, amp: bool, dev) -> list[float]:
    """Train on this one batch only: AdamW, no weight decay, no schedule; the loss before every step."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        with autocast(dev, amp):
            logit = model(x)
        loss = loss_fn(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description="Checks before training")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=16, help="photographs in the batch")
    ap.add_argument("--steps", type=int, default=100, help="steps of the overfitting check")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    cfg = load_config(a.config) if a.config else load_config()
    amp = cfg["train"]["amp"]
    dev = device()
    seed_everything(cfg["seed"])
    x, y, prior = one_batch(cfg, a.n)
    x, y = x.to(dev), y.to(dev)
    print(f"{a.n} photographs at {cfg['data']['image_size']} px, on {dev.type}")
    failed = []

    def report(ok: bool, check: str, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
        if not ok:
            failed.append(check)

    # 1. One forward pass, in eval mode as predict.py runs it.
    model = fresh(cfg, prior, x, dev)
    model.eval()
    with torch.no_grad(), autocast(dev, amp):
        out = model(x)
    p = to_prediction(out)
    report(out.shape == (a.n,) and bool(torch.isfinite(out).all()) and float(p.min()) >= 0 and float(p.max()) <= 1,
           "one forward pass", f"output shape {tuple(out.shape)}, predictions {float(p.min()):.3f} to {float(p.max()):.3f}")

    # 2. The loss before training against the loss of predicting the training mean.
    with torch.no_grad():
        untrained = float(loss_fn(out, y))
    constant = float(((y - prior) ** 2).mean())
    report(untrained <= 1.25 * constant + 1e-4, "the loss before training",
           f"{untrained:.4f}, the training mean scores {constant:.4f} (should be close)")

    # 3. Overfit one batch.
    start = time.time()
    losses = train_steps(model, x, y, a.steps, a.lr, amp, dev)
    report(losses[-1] < 1e-3, "overfit one batch",
           f"loss {losses[0]:.4f} -> {losses[-1]:.5f} in {a.steps} steps, {time.time() - start:.0f} s "
           f"(should end below 0.001)")

    # 4. The same seed twice: five steps each, from the same seed.
    runs = []
    for _ in range(2):
        seed_everything(cfg["seed"])
        runs.append(train_steps(fresh(cfg, prior, x, dev), x, y, 5, a.lr, amp, dev))
    gap = float(np.max(np.abs(np.subtract(*runs))))
    tol = 1e-6 if dev.type == "cpu" else 1e-4     # a GPU may add up in a different order between runs
    report(gap <= tol, "the same seed twice", f"largest difference of the losses {gap:.2e} (tolerance {tol:.0e})")

    print("all checks passed" if not failed else f"{len(failed)} failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
