"""The data: the tables, the folds, the resized copies, and the photographs as the network sees them.

Anything that touches pixels lives here and nowhere else. If training and
prediction prepared an image in two slightly different ways, the difference
would not show in the validation score and would show on the leaderboard.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET = "usability"
GREY = 128


# ------------------------------------------------------------------ tables

def train_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["data"]["raw"]) / "train.csv")


def test_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["data"]["raw"]) / "sample_submission.csv")[["image_id"]]


def assign_folds(df: pd.DataFrame, column: str, n_folds: int) -> np.ndarray:
    """A fold number per row; every unit of `column` lands whole in one fold.

    Units are dealt largest first to the fold with the fewest photographs so
    far, which keeps the folds close in size. Deterministic: the same table
    gives the same folds on every machine.
    """
    sizes = df[column].value_counts().sort_index()
    order = sorted(sizes.index, key=lambda u: (-sizes[u], u))
    load, fold_of = [0] * n_folds, {}
    for u in order:
        k = int(np.argmin(load))
        fold_of[u] = k
        load[k] += sizes[u]
    return df[column].map(fold_of).to_numpy()


# ------------------------------------------------------------------ images

def cache_path(cfg: dict, split: str, image_id: str) -> Path:
    return Path(cfg["data"]["work"]) / f"{split}_{cfg['data']['cache_size']}" / f"{image_id}.jpg"


def resize_photo(src: Path, dst: Path, size: int) -> None:
    """Long side to `size` px, aspect kept. Used for training and test photographs alike."""
    im = Image.open(src)
    im.draft("RGB", (size, size))          # decode the JPEG at 1/2 or 1/4 scale when that is enough
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=95)


def letterbox(path: Path, size: int) -> np.ndarray:
    """The photograph as grey uint8 (size, size): long side scaled to `size`, centred on mid-grey.

    Area averaging when shrinking, so that fine blur survives the resize as
    well as it can; that blur is what the label is mostly about. Grey, as in
    labs 3 and 4: the cache keeps the colour for whoever needs it.
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    s = size / max(h, w)
    if s != 1.0:
        im = cv2.resize(im, (round(w * s), round(h * s)),
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = im.shape
    out = np.full((size, size), GREY, np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    out[top:top + h, left:left + w] = im
    return out


class Photos(Dataset):
    """Photographs from the resize cache, letterboxed, as float tensors (1, size, size) in [0, 1].

    With `augment`, each photograph is mirrored left-right with probability
    1/2 and shifted by up to `max_shift` px (lab 4's "safe" augmentation).
    Nothing else: brightness, contrast and blur are what the label measures,
    so changing them would make a training example with the wrong label.
    """

    def __init__(self, paths, targets=None, size: int = 768, augment: bool = False,
                 max_shift: int = 16):
        self.paths = list(paths)
        self.targets = None if targets is None else np.asarray(targets, np.float32)
        self.size, self.augment, self.max_shift = size, augment, max_shift

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = letterbox(self.paths[i], self.size)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            s = self.max_shift
            if s:
                pad = np.pad(x, s, constant_values=GREY)
                dy, dx = np.random.randint(0, 2 * s + 1, size=2)
                x = pad[dy:dy + self.size, dx:dx + self.size]
        x = torch.from_numpy(np.ascontiguousarray(x)[None]).float().div_(255)
        y = torch.tensor(self.targets[i]) if self.targets is not None else torch.tensor(float("nan"))
        return x, y


def worker_init(worker_id: int) -> None:
    # each DataLoader worker gets its own numpy stream, derived from torch's seed
    np.random.seed(torch.initial_seed() % 2**32)
