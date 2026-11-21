"""Predict the test photographs and write a submission.

    python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
    python -m src.predict a.pt b.pt c.pt --out submissions/ensemble.csv
    python -m src.predict --ensemble --config configs/effnet_b0_ensemble.yaml --models models/effnet_b0_ensemble --tta

With several checkpoints the predictions are averaged. Every photograph is
predicted on its own; nothing about one test image influences another (rule
"one image in, one prediction out").

--ensemble finds the checkpoints itself instead of you listing them: the fold
models that `src.train --fold 0..n-1 --config CONFIG --out MODELS` wrote, which
are named after CONFIG (see src/config.py). Their predictions are averaged.
--with-all adds the model of `--fold all` as one more member. Every checkpoint
must exist, otherwise it stops and says which are missing.

--tta predicts every photograph twice, as it is and mirrored left-right, and
averages the two. It works with one checkpoint and with --ensemble.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.config import experiment_name, load_config, run_name
from src.data import TARGET, Photos, cache_path, test_table
from src.evaluate import predict
from src.model import load_checkpoint


def ensemble_checkpoints(a) -> tuple[dict, str, list[Path]]:
    """The config, the experiment's name and the checkpoints of its fold models (and the all-data model)."""
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    exp = experiment_name(cfg, a.config, a.name)
    folder = Path(a.models) if a.models else Path("models") / exp
    runs = list(range(cfg["split"]["n_folds"])) + (["all"] if a.with_all else [])
    paths = [folder / f"{run_name(cfg, a.config, k, a.name)}.pt" for k in runs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("these checkpoints are missing (train them with src.train --fold K --config ... --out ...):\n  "
                         + "\n  ".join(map(str, missing)))
    return cfg, exp, paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a submission from one or more checkpoints")
    ap.add_argument("checkpoints", nargs="*", help="checkpoints to average (not with --ensemble)")
    ap.add_argument("--out", default=None,
                    help="default submissions/reference.csv; with --ensemble submissions/<experiment>[_all][_tta].csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="average each prediction with that of the mirrored photograph")
    ap.add_argument("--ensemble", action="store_true",
                    help="average the fold models of an experiment, found through --config and --models")
    ap.add_argument("--config", default=None, help="with --ensemble: the config the fold models were trained with")
    ap.add_argument("--models", default=None,
                    help="with --ensemble: the folder the fold models are in (the --out of src.train); "
                         "default models/<experiment>")
    ap.add_argument("--name", default=None, help="with --ensemble: the --name the fold models were trained with")
    ap.add_argument("--seed", type=int, default=None, help="with --ensemble: the --seed they were trained with")
    ap.add_argument("--with-all", action="store_true", help="with --ensemble: also average in the --fold all model")
    a = ap.parse_args()

    ref_cfg = None
    if a.ensemble:
        if a.checkpoints:
            ap.error("--ensemble finds the checkpoints itself; do not list any")
        if not a.config:
            ap.error("--ensemble needs --config, the config the fold models were trained with")
        ref_cfg, exp, checkpoints = ensemble_checkpoints(a)
        out = Path(a.out or f"submissions/{exp}{'_all' if a.with_all else ''}{'_tta' if a.tta else ''}.csv")
    else:
        if not a.checkpoints:
            ap.error("give at least one checkpoint, or use --ensemble")
        if a.config or a.models or a.name or a.seed is not None or a.with_all:
            ap.error("--config, --models, --name, --seed and --with-all belong to --ensemble")
        checkpoints, out = a.checkpoints, Path(a.out or "submissions/reference.csv")

    preds, sub = [], None
    for ck in checkpoints:
        model, cfg = load_checkpoint(ck)
        if ref_cfg is not None and cfg != ref_cfg:
            print(f"WARNING: {ck} was trained with a different config than {a.config}")
        sub = test_table(cfg)
        paths = [cache_path(cfg, "test", i) for i in sub["image_id"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} test photographs are not resized yet; "
                             f"run python -m src.prepare first")
        dl = DataLoader(Photos(paths, size=cfg["data"]["image_size"]),
                        batch_size=a.batch_size, num_workers=a.workers)
        preds.append(predict(model, dl, cfg["train"]["amp"], tta=a.tta))
        print(f"{ck}: mean {preds[-1].mean():.4f}, sd {preds[-1].std():.4f}")
        del model

    sub[TARGET] = np.clip(np.mean(preds, axis=0), 0, 1).round(5)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"{len(preds)} model(s){', mirror TTA' if a.tta else ''}; {len(sub)} rows -> {out}")


if __name__ == "__main__":
    main()
