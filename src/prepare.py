"""Resize every training and test photograph once, into data/work/reference/.

Decoding a 2048 px JPEG costs more than the network's forward pass at 768 px,
so training reads smaller copies. Training and test photographs go through the
same function. Idempotent: files already there are skipped.

    python -m src.prepare
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import load_config
from src.data import cache_path, resize_photo, test_table, train_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    raw, size = Path(cfg["data"]["raw"]), cfg["data"]["cache_size"]

    jobs = []
    for split, df in (("train", train_table(cfg)), ("test", test_table(cfg))):
        missing = [i for i in df["image_id"] if not (raw / split / f"{i}.jpg").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} {split} photographs are missing from {raw / split}, "
                             f"e.g. {missing[0]}.jpg")
        todo = [(raw / split / f"{i}.jpg", cache_path(cfg, split, i), size)
                for i in df["image_id"] if not cache_path(cfg, split, i).exists()]
        print(f"{split}: {len(df)} photographs, {len(todo)} to resize")
        jobs += todo

    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(lambda j: resize_photo(*j), jobs), 1):
            if n % 250 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print(f"done: {Path(cfg['data']['work'])}")


if __name__ == "__main__":
    main()
