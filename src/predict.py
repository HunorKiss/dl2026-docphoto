"""Predict the test photographs and write a submission.

    python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
    python -m src.predict a.pt b.pt c.pt --out submissions/ensemble.csv

With several checkpoints the predictions are averaged. Every photograph is
predicted on its own; nothing about one test image influences another (rule
"one image in, one prediction out").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.data import TARGET, Photos, cache_path, test_table
from src.evaluate import predict
from src.model import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a submission from one or more checkpoints")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default="submissions/reference.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    preds, sub = [], None
    for ck in a.checkpoints:
        model, cfg = load_checkpoint(ck)
        sub = test_table(cfg)
        paths = [cache_path(cfg, "test", i) for i in sub["image_id"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} test photographs are not resized yet; "
                             f"run python -m src.prepare first")
        dl = DataLoader(Photos(paths, size=cfg["data"]["image_size"]),
                        batch_size=a.batch_size, num_workers=a.workers)
        preds.append(predict(model, dl, cfg["train"]["amp"]))
        print(f"{ck}: mean {preds[-1].mean():.4f}, sd {preds[-1].std():.4f}")

    sub[TARGET] = np.clip(np.mean(preds, axis=0), 0, 1).round(5)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"{len(sub)} rows -> {out}")


if __name__ == "__main__":
    main()
