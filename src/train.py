"""Train lab 4's network to predict usability.

    python -m src.train --fold 0      # train on folds 1-3, report RMSE on fold 0
    python -m src.train --fold all    # train on every photograph: the model you submit
    python -m src.train --fold 0 --tta   # validate with mirror test-time augmentation
    python -m src.train --fold all --epochs 12   # override train.epochs of the config

The checkpoints are named after the config file, <config>_fold<k>_seed<s>.pt (see
src/config.py). To average the four fold models into one submission, use
`python -m src.predict --ensemble`.

A fold run writes its checkpoint and its validation predictions; the `all` run
writes only the checkpoint, since nothing is left to validate on. Both use the
same number of epochs, the one in the config, so what you measured on a fold is
what you get from the final run. Each run prints the line to put in your notes.

What one run does, in order:
  1. reads the config and fixes the random seed,
  2. reads train.csv and deals the photographs into folds by `group_strict`,
  3. builds the data loaders from the resized copies that prepare.py wrote,
  4. builds the network and measures the pixel statistics it standardises with,
  5. trains for a fixed number of epochs with AdamW and a warm-up-cosine schedule,
  6. saves the checkpoint, the per-epoch log and (fold runs only) the
     validation predictions.
Every run is also logged to Weights & Biases (src/tracking.py); WANDB_MODE=disabled
in .env turns that off.

This file only puts the pieces together, each in a file of its own:
  src/config.py    the config, the seed, the device, mixed precision
  src/data.py      the tables, the folds, the images, the augmentation
  src/model.py     the network
  src/loss.py      the loss, and how the network's output becomes a prediction
  src/optim.py     the parameter groups and the learning-rate schedule
  src/evaluate.py  predictions for a loader, and the RMSE
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import tracking
from src.config import amp_dtype, autocast, device, load_config, run_name, seed_everything
from src.data import TARGET, Photos, assign_folds, cache_path, train_table, worker_init
from src.evaluate import predict, rmse
from src.loss import make_loss, to_prediction
from src.model import build_model
from src.optim import lr_at, param_groups


class EarlyStopping:
    """Stop when val_rmse has not improved for `patience` epochs in a row (train.early_stopping).

    The model of the epoch it stops at is kept, as the model of the last epoch
    is kept without early stopping, never the best one: choosing the epoch on
    the fold you report would flatter the score (see section 6 below).
    """

    def __init__(self, patience: int):
        self.patience, self.best, self.bad = int(patience), float("inf"), 0

    def step(self, val: float) -> bool:
        """Record one epoch's validation RMSE; True when training should stop."""
        if val < self.best:
            self.best, self.bad = val, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def rescale_phase_epochs(phase_epochs: list[int], n: int) -> list[int]:
    """Rescale `phase_epochs` proportionally so that they sum to `n` (for --epochs).

    Rounding rule: every phase gets floor(e * n / sum(e)) epochs, but at least 1.
    Then single epochs are added to the phases furthest below their exact share
    (largest shortfall first), or removed from the phases furthest above it
    (only from phases with more than 1 epoch), until the sum is exactly `n`;
    ties go to the earlier phase. If `n` already equals the sum, nothing changes.
    """
    k = len(phase_epochs)
    if n < k:
        raise ValueError(f"--epochs {n} is fewer than the {k} phases {phase_epochs}; every phase needs at least 1 epoch")
    total = sum(phase_epochs)
    ideal = [e * n / total for e in phase_epochs]
    out = [max(1, math.floor(x)) for x in ideal]
    while sum(out) < n:
        i = max(range(k), key=lambda j: (ideal[j] - out[j], -j))
        out[i] += 1
    while sum(out) > n:
        i = max((j for j in range(k) if out[j] > 1), key=lambda j: (out[j] - ideal[j], -j))
        out[i] -= 1
    return out


def pixel_stats(ds: Photos, n: int = 200) -> tuple[float, float]:
    """Mean and standard deviation of the pixels of up to `n` training photographs.

    The photographs are spread evenly over the table, not the first `n`, which
    would all come from the same few photographers. The letterbox's grey border
    is included, as it is in every image the network sees.
    """
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    x = torch.stack([ds[i][0] for i in idx])
    return float(x.mean()), float(x.std())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the reference model")
    ap.add_argument("--fold", required=True, help="a fold number to validate on, or 'all'")
    ap.add_argument("--tta", action="store_true",
                    help="validate with mirror test-time augmentation: every validation photograph is "
                         "predicted as it is and mirrored, and the two predictions are averaged")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, choices=["small_cnn", "effnet_b0", "convnext_tiny"],
                    help="override the config's model.type (small_cnn | effnet_b0 | convnext_tiny)")
    ap.add_argument("--name", default=None, help="override the run name (W&B and output files)")
    ap.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    ap.add_argument("--epochs", type=int, default=None,
                    help="overrides the config's train.epochs; the schedule and the phases (phase_epochs "
                         "rescaled proportionally) are computed from it")
    ap.add_argument("--out", default="models/reference")
    a = ap.parse_args()

    # --- 1. Config and seed. Every number that changes the result comes from
    # the config; --seed exists only to measure how much the score moves
    # between seeds (a rerun with another seed is not an experiment step).
    cfg = load_config(a.config) if a.config else load_config()
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.model is not None:
        cfg.setdefault("model", {})["type"] = a.model    # command-line choice of network
    d, t = cfg["data"], cfg["train"]
    if a.epochs is not None:
        if a.epochs < 1:
            raise ValueError(f"--epochs must be at least 1, got {a.epochs}")
        t["epochs"] = a.epochs                       # the schedule is built from this
        if t.get("phases") is not None and t.get("phase_epochs") is not None:
            t["phase_epochs"] = rescale_phase_epochs([int(e) for e in t["phase_epochs"]], a.epochs)
            print(f"--epochs {a.epochs}: phase_epochs rescaled to {t['phase_epochs']}")
    # The run's name ends up in W&B and in every file it writes (checkpoint,
    # log, validation predictions). It is <experiment>_fold<k>_seed<s>, where the
    # experiment is, in order of preference: --name; wandb.experiment in the
    # config; or the config file's own name (configs/effnet_b0_dropout.yaml ->
    # "effnet_b0_dropout"). See src/config.py.
    name = run_name(cfg, a.config, a.fold, a.name)
    # Weights & Biases, before the seed is set: whatever the logging does, it
    # cannot change a random number the training draws. The --out folder is
    # the group, so the four fold runs of one experiment sit together.
    tracking.start({**cfg, "fold": a.fold, "out": a.out, "val_tta": a.tta}, name, group=Path(a.out).name,
                   job_type="all" if a.fold == "all" else "fold")
    seed_everything(cfg["seed"])

    # --- 2. Folds. Every unit of `group_strict` lands whole in one fold, so no
    # page and no layout is on both sides of the split. The assignment is
    # deterministic: the same train.csv gives the same folds on every machine,
    # which is what makes fold scores comparable between your runs.
    df = train_table(cfg)
    df["fold"] = assign_folds(df, cfg["split"]["column"], cfg["split"]["n_folds"])
    if a.fold == "all":
        tr, va = df, df.iloc[:0]                    # everything trains, nothing validates
    else:
        k = int(a.fold)
        tr, va = df[df["fold"] != k], df[df["fold"] == k]
    print(f"{name}: {len(tr)} training photographs, {len(va)} validation")

    # --- 3. Data loaders. They read the 1024 px copies written by prepare.py
    # (run it first) and letterbox them to image_size grey squares. Only the
    # training loader augments and shuffles; it also drops the last, smaller
    # batch, so that every step's gradient is averaged over the same number of
    # photographs. The generator seeds the shuffling, worker_init the
    # augmentation in each worker process.
    def loader(rows, augment):
        ds = Photos([cache_path(cfg, "train", i) for i in rows["image_id"]], rows[TARGET],
                    size=d["image_size"], augment=augment, max_shift=t["max_shift"])
        g = torch.Generator().manual_seed(cfg["seed"])
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=augment, drop_last=augment,
                          num_workers=t["num_workers"], pin_memory=True, generator=g,
                          worker_init_fn=worker_init, persistent_workers=t["num_workers"] > 0)

    train_dl = loader(tr, augment=True)
    val_dl = loader(va, augment=False) if len(va) else None

    # --- 4. The network. `prior` sets the output bias so that the untrained
    # network already predicts the training mean. The input statistics are
    # measured on the training photographs of this run only (never on the
    # validation fold) and stored inside the model, so they travel with the
    # checkpoint and predict.py applies exactly the same standardisation.
    dev = device()
    # Asynchronous host-to-device copies are for CUDA with pinned memory. On MPS
    # pinned memory does not exist, and an asynchronous copy from ordinary memory
    # can read a batch that has already been overwritten.
    nb = dev.type == "cuda"
    model = build_model(cfg, prior=float(tr[TARGET].mean()))
    mean, std = pixel_stats(Photos([cache_path(cfg, "train", i) for i in tr["image_id"]],
                                   size=d["image_size"]))
    model.in_mean.fill_(mean)
    model.in_std.fill_(std)
    model.to(dev)

    # --- 5. Training phases, optimiser and schedule.
    # A "phase" trains either the head only ("head", backbone frozen) or the
    # whole network ("all", backbone unfrozen), for a number of epochs and at
    # its own learning rate. The config lists the phases in order:
    #   phases: [head, all]   head first, then fine-tune the whole net (the usual order)
    #   phases: [all, head]   the whole net first, then the head
    #   phases: [head]        head only, for all epochs
    #   phases: [all]         the whole net only, for all epochs
    # Learning rate per phase: "head" uses train.lr, "all" uses train.finetune_lr
    # (default lr/10) -- the pretrained backbone is nudged gently so fine-tuning
    # does not overwrite what ImageNet taught it.
    # Epochs per phase: give `phase_epochs` (a list parallel to `phases`, e.g.
    # [4, 8]) to set them explicitly; otherwise `epochs` is split evenly across
    # the phases. Backward compatibility: if no `phases` is given but
    # `freeze_epochs` is, that maps to [head, all] with freeze_epochs on the head.
    head_lr = t["lr"]
    finetune_lr = float(t.get("finetune_lr", t["lr"] / 10))
    can_freeze = hasattr(model, "set_backbone_frozen")

    phase_names = t.get("phases")
    phase_epochs = t.get("phase_epochs")
    # A config with neither `phases` nor `freeze_epochs` (configs/reference.yaml)
    # is a plain run: the whole network at train.lr, as before phases existed.
    plain_run = phase_names is None and not (can_freeze and int(t.get("freeze_epochs", 0)) > 0)
    if phase_names is None:
        fe = int(t.get("freeze_epochs", 0))
        phase_names = ["head", "all"] if (can_freeze and fe > 0) else ["all"]
        split = [fe, t["epochs"] - fe] if phase_names == ["head", "all"] else [t["epochs"]]
    elif phase_epochs is not None:
        # explicit epochs per phase
        if len(phase_epochs) != len(phase_names):
            raise ValueError(f"phase_epochs {phase_epochs} must have one entry per phase {phase_names}")
        split = [int(e) for e in phase_epochs]
        if sum(split) != t["epochs"]:
            print(f"note: phase_epochs sum {sum(split)} != epochs {t['epochs']}; using the phase_epochs sum")
    else:
        # split the epochs across the listed phases as evenly as possible
        n = len(phase_names)
        base, extra = divmod(t["epochs"], n)
        split = [base + (1 if i < extra else 0) for i in range(n)]

    phases = []           # (mode, n_epochs, lr), skipping zero-epoch phases
    # `phase`, not `name`: `name` is the run's name and the checkpoint is saved under it.
    for phase, ne in zip(phase_names, split, strict=True):
        if ne <= 0:
            continue
        if phase not in ("head", "all"):
            raise ValueError(f"unknown phase {phase!r}; use 'head' or 'all'")
        phases.append((phase, ne, head_lr if (phase == "head" or plain_run) else finetune_lr))
    print("training phases:", [(m, e, f"lr {lr:.1e}") for m, e, lr in phases])

    # Optional: in "all" phases the head gets its own, larger learning rate,
    # train.finetune_head_lr, while the backbone trains at finetune_lr (for
    # example backbone 1e-5, head 1e-4). Without it every parameter of an "all"
    # phase uses finetune_lr, as before.
    finetune_head_lr = t.get("finetune_head_lr")
    if finetune_head_lr is not None:
        if not hasattr(model, "head_parameters"):
            raise ValueError(f"train.finetune_head_lr needs a model with a separate head; "
                             f"{cfg['model'].get('type', 'small_cnn')!r} has none")
        finetune_head_lr = float(finetune_head_lr)
        print(f"'all' phases: backbone at finetune_lr {finetune_lr:.1e}, head at finetune_head_lr {finetune_head_lr:.1e}")

    # Gradient accumulation (train.accum_steps, default 1): the gradients of
    # accum_steps batches are summed before one optimiser step, so batch 4 with
    # accum 2 updates like batch 8 at the memory of batch 4. The schedule counts
    # optimiser steps, not batches.
    accum = int(t.get("accum_steps", 1))
    n_batches = len(train_dl)
    steps_per_epoch = math.ceil(n_batches / accum)
    criterion = make_loss(cfg)
    # float16 needs loss scaling, or small gradients round to zero; with any
    # other precision the scaler is switched off and does nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(dev, t["amp"]) == torch.float16)
    if accum > 1 or t.get("loss", "mse") != "mse" or scaler.is_enabled():
        print(f"batch {t['batch_size']} x accum {accum} = {t['batch_size'] * accum} per step | "
              f"loss {t.get('loss', 'mse')} | fp16 loss scaling {scaler.is_enabled()}")

    def make_opt_sched(epochs_left: int, lr: float, head_lr_: float | None = None):
        groups = (param_groups(model, t["weight_decay"], model.head_parameters(), head_lr_) if head_lr_ is not None
                  else param_groups(model, t["weight_decay"]))
        o = torch.optim.AdamW(groups, lr=lr)
        tot = max(1, epochs_left) * steps_per_epoch
        s = torch.optim.lr_scheduler.LambdaLR(o, lambda step: lr_at(step, tot, t["warmup"]))
        return o, s

    def apply_phase(mode: str, n_epochs: int, lr: float):
        """Set head-only vs whole-network, and build a fresh optimiser/schedule for this phase."""
        if can_freeze:
            model.set_backbone_frozen(mode == "head")      # "head" -> frozen backbone; "all" -> unfrozen
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        split_lr = finetune_head_lr if (mode == "all" and not plain_run) else None
        lr_text = f"lr {lr:.1e}" + (f" (head {split_lr:.1e})" if split_lr is not None else "")
        print(f"  phase '{mode}': {n_epochs} epochs at {lr_text}, {n_trainable:,} trainable parameters")
        head_split[0] = split_lr is not None
        return make_opt_sched(n_epochs, lr, split_lr)

    head_split = [False]    # whether the current phase gives the head its own lr (its groups come last)

    # boundaries: the epoch (1-based) at which each phase begins
    phase_start, acc = [], 1
    for _, ne, _ in phases:
        phase_start.append(acc)
        acc += ne
    phase_of_start = {start: idx for idx, start in enumerate(phase_start)}

    opt, sched = apply_phase(*phases[0])

    total_epochs = sum(ne for _, ne, _ in phases)   # the real epoch count (phase_epochs may override `epochs`)
    # Early stopping (train.early_stopping: patience in epochs) needs a
    # validation fold, so it applies to fold runs only; `--fold all` always
    # trains every epoch.
    stopper = EarlyStopping(t["early_stopping"]) if t.get("early_stopping") and val_dl is not None else None
    stopped_epoch = None
    log = []
    for epoch in range(1, total_epochs + 1):
        if epoch in phase_of_start and epoch != 1:         # a new phase begins this epoch
            opt, sched = apply_phase(*phases[phase_of_start[epoch]])
        model.train()                               # BatchNorm uses batch statistics
        start, losses, sq_err, seen = time.time(), [], 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=nb), y.to(dev, non_blocking=nb)
            # The forward pass in reduced precision where train.amp asks for it
            # and the GPU supports it (faster, half the memory); the loss in full precision.
            with autocast(dev, t["amp"]):
                logit = model(x)
            # The loss is computed on the sigmoid of the logit (src/loss.py): the
            # prediction stays in [0, 1] like the label.
            loss = criterion(logit, y)
            # Each batch of an accumulation group contributes its share, so the
            # summed gradient is that of the group's mean loss (the last group
            # of an epoch may be shorter).
            group = min(accum, n_batches - (i // accum) * accum)
            scaler.scale(loss / group).backward()
            if (i + 1) % accum == 0 or i + 1 == n_batches:
                # Clipping the gradient norm at 1 stops a single bad batch from
                # making a huge step; in a normal step it does nothing. With
                # fp16 the gradients are unscaled first, so the clip sees their true size.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            with torch.no_grad():
                sq_err += float(((to_prediction(logit) - y) ** 2).sum())
                seen += len(y)
        # train_rmse is measured on augmented photographs while the weights were
        # still changing, so it is only a rough guide; val_rmse (fold runs) is
        # computed after the epoch on every photograph of the held-out fold.
        # train_loss is the training loss itself (with MSE: train_rmse squared).
        row = {"epoch": epoch, "train_rmse": math.sqrt(sq_err / seen), "train_loss": float(np.mean(losses)),
               "seconds": round(time.time() - start)}
        if val_dl is not None:
            row["val_rmse"] = rmse(va[TARGET], predict(model, val_dl, t["amp"], tta=a.tta))
        log.append(row)
        lrs = sched.get_last_lr()
        tracking.log({**row, "lr": lrs[0], **({"lr_head": lrs[-1]} if head_split[0] else {})})
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in row.items()), flush=True)
        if stopper is not None and stopper.step(row["val_rmse"]):
            stopped_epoch = epoch
            print(f"  early stopping: val_rmse has not improved for {stopper.patience} epochs "
                  f"(best {stopper.best:.4f}); keeping the model of epoch {epoch}")
            break

    # --- 6. Outputs. The model after the last epoch is kept, never the epoch
    # with the best validation score: picking the best epoch on the fold you
    # report would flatter the score, and the `all` run has no fold to pick on.
    # The config goes into the checkpoint, so predict.py rebuilds the same
    # network and image size from the checkpoint alone.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict(), "fold": a.fold},
               out / f"{name}.pt")
    pd.DataFrame(log).to_csv(out / f"{name}_log.csv", index=False)
    summary = {}
    if val_dl is not None:
        # One row per validation photograph: label, prediction and the columns
        # you need for error analysis. Pool the four folds' files for the
        # out-of-fold RMSE in EXPERIMENTS.md.
        p = predict(model, val_dl, t["amp"], tta=a.tta)
        res = va[["image_id", TARGET, "eval_ok", "family"]].assign(pred=p)
        res.to_csv(out / f"{name}_val.csv", index=False)
        # The reported score leaves out the 47 photographs with eval_ok = 0,
        # uploaded heavily downscaled (fine for training, not for measuring;
        # lab 3). The per-epoch val_rmse above counts them all, so the two
        # differ slightly.
        ok = res["eval_ok"] == 1
        summary = {"val_rmse_eval_ok": rmse(res.loc[ok, TARGET], res.loc[ok, "pred"]), "n_eval_ok": int(ok.sum()),
                   "epochs_trained": len(log), **({"stopped_epoch": stopped_epoch} if stopped_epoch else {})}
        print(f"{name}: validation RMSE {summary['val_rmse_eval_ok']:.4f} "
              f"on {summary['n_eval_ok']} photographs (eval_ok){', mirror TTA' if a.tta else ''}")
    tracking.finish(summary)
    print(f"-> {out / (name + '.pt')}")


if __name__ == "__main__":
    main()
