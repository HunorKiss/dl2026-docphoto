"""Experiment tracking with Weights & Biases, as in labs 3 and 4.

Every run of train.py logs its config, its curves (training and validation
RMSE and the learning rate, per epoch) and its final validation RMSE. The four
fold runs of one experiment share a group named after their --out folder, so
they sit together on the W&B page, and the experiments of six weeks can be
compared there. EXPERIMENTS.md stays the record you hand in; W&B is where
you look things up.

Setup, once: paste your key from https://wandb.ai/authorize into .env after
WANDB_API_KEY=. Docker Compose passes it, and WANDB_MODE, into the container:

    WANDB_MODE=online      log to wandb.ai (the default)
    WANDB_MODE=offline     log into wandb/ only; upload later with `wandb sync wandb/offline-run-*`
    WANDB_MODE=disabled    no logging at all

No key is also fine: the run is then logged anonymously, with a link you can
claim later. If wandb.ai cannot be reached, the run falls back to offline and
says so; logging never stops a training run.

Logging changes nothing in the training: train.py starts the run before it
sets the seed, and nothing here draws a random number.
"""
from __future__ import annotations

import os

import wandb

from src.config import ROOT


def start(cfg: dict, name: str, group: str, job_type: str):
    """Start a W&B run; returns it, or None when logging is disabled or cannot start at all."""
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "disabled":
        return None
    kwargs = dict(project=cfg.get("wandb", {}).get("project", "snap-or-retake"), name=name, group=group,
                  job_type=job_type, config=cfg, mode=mode, dir=str(ROOT), reinit=True)
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        kwargs["anonymous"] = "allow"   # no key: an anonymous run with a claimable link
    try:
        return wandb.init(**kwargs)
    except Exception as e:  # noqa: BLE001  (anything: no network, bad key, service down)
        print(f"wandb: could not start an online run ({type(e).__name__}: {e}); logging offline instead")
        kwargs.pop("anonymous", None)
        kwargs["mode"] = "offline"
        try:
            return wandb.init(**kwargs)
        except Exception as e2:  # noqa: BLE001
            print(f"wandb: no logging for this run ({type(e2).__name__}: {e2})")
            return None


def log(row: dict) -> None:
    """One epoch's numbers, with the epoch as the step."""
    if wandb.run is not None:
        wandb.log(row, step=row["epoch"])


def finish(summary: dict | None = None) -> None:
    """The run's final numbers into its summary, then close it."""
    if wandb.run is not None:
        wandb.run.summary.update(summary or {})
        wandb.finish()
