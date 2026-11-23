"""Zero-shot pre-test: how well does a pretrained MUSIQ quality score predict `usability`?

No training. Every training photograph is scored by MUSIQ (KonIQ-10k and SPAQ
weights, grey and RGB input); the script reports the Spearman and Pearson
correlation of the raw score with `usability`, and the RMSE of a score ->
usability mapping fitted out-of-fold on the repo's group_strict folds, so the
number is comparable to our OOF RMSEs (reference 0.1956, EfficientNet-B0 ~0.145).

Mapping: a linear fit (2 parameters, clipped to [0, 1]) is the headline: it
cannot overfit and does not care about the arbitrary scale of the MOS. An
isotonic fit (monotone, the ceiling for any monotone mapping) is the second column.
Each is fitted on three folds and applied to the fourth.

Kaggle cell (the setup cell has cloned the repo, linked the data, run prepare;
needs Internet on for the two weight downloads, a few minutes on a T4):

    import os
    os.chdir("/kaggle/working/repo")
    os.environ.update(PYTHONPATH="/kaggle/working/repo", WANDB_MODE="disabled",
                      MUSIQ_WEIGHTS_DIR="/kaggle/temp/musiq")
    assert os.path.isdir("data/work/reference/train_1024"), "run python -m src.prepare first"
    !python scripts/musiq_pretest.py --weights koniq spaq --input both --batch-size 8 --workers 2

Locally: env PYTHONPATH=. python3 scripts/musiq_pretest.py --limit 64 --workers 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import device, load_config                                   # noqa: E402
from src.data import MultiscalePhotos, TARGET, assign_folds, cache_path, train_table  # noqa: E402
from src.musiq import load_musiq                                             # noqa: E402


def oof_map(score, y, folds, kind: str = "linear") -> np.ndarray:
    """Out-of-fold usability from the score: each fold predicted by a fit on the other folds."""
    score, y, folds = np.asarray(score, float), np.asarray(y, float), np.asarray(folds)
    out = np.zeros_like(y)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if kind == "linear":
            a, b = np.polyfit(score[tr], y[tr], 1)
            out[te] = a * score[te] + b
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            out[te] = iso.fit(score[tr], y[tr]).predict(score[te])
    return np.clip(out, 0, 1)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@torch.no_grad()
def score_photos(model, paths, color, batch_size, workers, dev, fp16) -> np.ndarray:
    ds = MultiscalePhotos(paths, canvas=1024, color=color)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    out = []
    for i, (x, _) in enumerate(dl):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            out.append(model(x.to(dev)).float().cpu())
        if (i + 1) % 20 == 0:
            print(f"  {min((i + 1) * batch_size, len(ds))}/{len(ds)}", flush=True)
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/reference.yaml")
    ap.add_argument("--weights", nargs="+", default=["koniq", "spaq"], choices=["koniq", "spaq"])
    ap.add_argument("--input", default="both", choices=["rgb", "grey", "both"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fp32", action="store_true", help="no fp16 autocast (default on CUDA)")
    ap.add_argument("--limit", type=int, default=0, help="first N rows, spread evenly (smoke tests only)")
    ap.add_argument("--out", default="models/musiq_pretest")
    a = ap.parse_args()

    cfg = load_config(a.config)
    df = train_table(cfg)
    df["fold"] = assign_folds(df, cfg["split"]["column"], cfg["split"]["n_folds"])
    if a.limit:
        df = df.iloc[np.linspace(0, len(df) - 1, a.limit).round().astype(int)].reset_index(drop=True)
    paths = [cache_path(cfg, "train", i) for i in df["image_id"]]
    if not all(p.exists() for p in paths):
        sys.exit("resized cache missing: run python -m src.prepare first")

    dev = device()
    fp16 = dev.type == "cuda" and not a.fp32
    inputs = ["rgb", "grey"] if a.input == "both" else [a.input]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y, folds, ok = df[TARGET].to_numpy(), df["fold"].to_numpy(), df["eval_ok"].to_numpy() == 1
    rows = []
    for w in a.weights:
        model = load_musiq(w, with_head=True).to(dev).eval()
        for inp in inputs:
            print(f"scoring {len(df)} photos: {w} / {inp} on {dev}", flush=True)
            s = score_photos(model, paths, inp == "rgb", a.batch_size, a.workers, dev, fp16)
            lin, iso = oof_map(s, y, folds, "linear"), oof_map(s, y, folds, "isotonic")
            pd.DataFrame({"image_id": df["image_id"], TARGET: y, "fold": folds,
                          "eval_ok": df["eval_ok"], "score": s}
                         ).to_csv(out / f"scores_{w}_{inp}.csv", index=False)
            rows.append({
                "weights": w, "input": inp,
                "spearman": spearmanr(s, y)[0], "pearson": pearsonr(s, y)[0],
                "spearman_ok": spearmanr(s[ok], y[ok])[0], "pearson_ok": pearsonr(s[ok], y[ok])[0],
                "rmse_linear_ok": rmse(lin[ok], y[ok]), "rmse_isotonic_ok": rmse(iso[ok], y[ok]),
                "rmse_linear_all": rmse(lin, y),
                "rmse_linear_folds": " ".join(f"{rmse(lin[folds == k], y[folds == k]):.4f}"
                                              for k in np.unique(folds)),
            })
        del model
    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"\nzero-shot MUSIQ vs {TARGET}, {len(df)} photos "
          f"({int(ok.sum())} with eval_ok), group_strict folds")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("reference RMSE 0.1956, EfficientNet-B0 ~0.145 (OOF, eval_ok rows)")


if __name__ == "__main__":
    main()
