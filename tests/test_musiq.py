"""Tests of src/musiq.py, MultiscalePhotos and the pre-test mapping.

    env PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE=disabled python3 -m pytest tests/test_musiq.py -q
    MUSIQ_NET=1 ... (also downloads the weights, ~216 MB, and checks the real model)
"""
from __future__ import annotations

import importlib.util
import math
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src import musiq
from src.data import MultiscalePhotos
from src.musiq import Musiq, build_tokens, native_size

ROOT = Path(__file__).resolve().parent.parent
PYIQA_UTIL = ("https://raw.githubusercontent.com/chaofengc/IQA-PyTorch/main/"
              "pyiqa/data/multiscale_trans_util.py")


def canvas_of(contents, size=256):
    """A canvas batch (B, C, size, size): each content (C, h, w) centred, the rest -1."""
    out = torch.full((len(contents), contents[0].shape[0], size, size), -1.0)
    for i, c in enumerate(contents):
        h, w = c.shape[-2:]
        top, left = (size - h) // 2, (size - w) // 2
        out[i, :, top:top + h, left:left + w] = c
    return out


def test_tokens_match_pyiqa(tmp_path):
    try:
        src = tmp_path / "util.py"
        urllib.request.urlretrieve(PYIQA_UTIL, src)
    except Exception:
        pytest.skip("cannot fetch the pyiqa reference")
    spec = importlib.util.spec_from_file_location("pyiqa_util", src)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch.manual_seed(0)
    img = torch.rand(1, 3, 320, 240) * 2 - 1
    r = ref.get_multiscale_patches(img, max_seq_len_from_original_res=-1)[0]
    r = r[r[:, -1] == 1]
    mine = musiq.sample_scales(img, native_max_patches=10**6, antialias=False)
    p = torch.cat([a.reshape(a.shape[0], -1) for a, _, _ in mine])
    pos = torch.cat([b for _, b, _ in mine])
    sc = torch.cat([torch.full_like(b, s) for _, b, s in mine])
    assert p.shape[0] == r.shape[0]
    assert torch.allclose(p, r[:, :-3], atol=1e-5)
    assert torch.equal(pos, r[:, -3].long()) and torch.equal(sc, r[:, -2].long())


@pytest.mark.parametrize("h,w,cap", [(1024, 768, 768), (768, 1024, 768), (1024, 1024, 768),
                                     (1024, 576, 768), (900, 1000, 300), (100, 50, 768)])
def test_native_size(h, w, cap):
    rh, rw = native_size(h, w, cap)
    assert math.ceil(rh / 32) * math.ceil(rw / 32) <= cap
    if math.ceil(h / 32) * math.ceil(w / 32) <= cap:
        assert (rh, rw) == (h, w)
    else:
        assert abs(rh / rw - h / w) * rw <= 1.5      # aspect kept within about 1 px


def test_mirror_and_batch_invariance():
    torch.manual_seed(0)
    m = Musiq(with_head=True).eval()
    a, b = torch.rand(1, 192, 128), torch.rand(1, 128, 192)
    with torch.no_grad():
        f = m.features(canvas_of([a]))
        assert torch.allclose(m.features(canvas_of([a]).flip(-1)),
                              m.features(canvas_of([a.flip(-1)])), atol=1e-5)
        both = m.features(canvas_of([a, b]))
        assert torch.allclose(f[0], both[0], atol=1e-4)
        assert torch.allclose(m.features(canvas_of([b]))[0], both[1], atol=1e-4)


def test_tokens_stem_only_valid():
    c = canvas_of([torch.rand(1, 192, 128), torch.rand(1, 128, 64)])
    patches, b_idx, slot, pos, scale, mask = build_tokens(c, 768)
    assert patches.shape[0] == int(mask.sum()) == len(b_idx) == len(slot)
    assert mask[:, 0].all() and mask.sum(1)[0] > mask.sum(1)[1]


@pytest.mark.skipif(not os.environ.get("MUSIQ_NET"), reason="set MUSIQ_NET=1 (downloads weights)")
def test_pretrained_sharp_beats_blur():
    m = musiq.load_musiq("koniq").eval()
    assert sum(p.numel() for p in m.parameters()) == 27_125_825
    files = sorted((ROOT / "data/work/reference/train_1024").glob("*.jpg"))
    if not files:
        pytest.skip("no cache")
    im = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(im, (0, 0), 8)
    xs = [torch.from_numpy(a).permute(2, 0, 1).float() / 255 for a in (im, blur)]
    with torch.no_grad():
        s = m(canvas_of(xs, 1024))
    assert s[0] > s[1]


# ------------------------------------------------------------ data

def test_multiscale_photos(tmp_path):
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), (np.random.rand(80, 60, 3) * 255).astype(np.uint8))
    ds = MultiscalePhotos([p], canvas=128)
    x, y = ds[0]
    assert x.shape == (1, 128, 128) and torch.isnan(y)
    valid = x[0] >= 0
    assert valid.sum() == 80 * 60
    assert valid[24:104, 34:94].all() and (x[0][~valid] == -1).all()
    assert x[0][valid].min() >= 0 and x[0][valid].max() <= 1
    assert MultiscalePhotos([p], canvas=128, color=True)[0][0].shape == (3, 128, 128)
    aug = MultiscalePhotos([p], [0.5], canvas=128, augment=True, max_shift=8)
    for _ in range(5):
        xa, ya = aug[0]
        assert xa.shape == (1, 128, 128) and float(ya) == 0.5
        assert (xa[0] >= 0).sum() >= 64 * 44
    x2 = ds[0][0]
    assert torch.equal(x2.flip(-1)[0] >= 0, x2[0] >= 0)       # centred: the box mirrors onto itself


# ------------------------------------------------------------ the pre-test mapping

def _pretest():
    spec = importlib.util.spec_from_file_location("musiq_pretest", ROOT / "scripts/musiq_pretest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oof_map_linear_and_isolated_folds():
    pt = _pretest()
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 100, 200)
    folds = np.arange(200) % 4
    y = 0.005 * s + 0.2
    assert pt.rmse(pt.oof_map(s, y, folds, "linear"), y) < 1e-9
    # a fold's own targets never enter its fit: corrupt one fold only, the others stay exact
    y2 = y.copy()
    y2[folds == 0] = 0.0
    pred = pt.oof_map(s, y2, folds, "linear")
    assert np.allclose(pred[folds == 0], y[folds == 0], atol=1e-9)
    assert pt.rmse(pred[folds != 0], y2[folds != 0]) > 0.01
    iso = pt.oof_map(s, y, folds, "isotonic")
    assert pt.rmse(iso, y) < 0.02 and iso.min() >= 0 and iso.max() <= 1
