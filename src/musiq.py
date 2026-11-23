"""MUSIQ (multi-scale image quality transformer), vendored, with a GPU token builder.

Source and licence
    Architecture and token layout are ported from IQA-PyTorch (pyiqa),
    https://github.com/chaofengc/IQA-PyTorch :
      pyiqa/archs/musiq_arch.py, pyiqa/data/multiscale_trans_util.py and
      exact_padding_2d from pyiqa/matlab_utils/padding.py
    (Copyright (c) 2022 Chaofeng Chen; PolyForm Noncommercial 1.0.0, earlier
    releases S-Lab 1.0). pyiqa re-implements Google's MUSIQ
    (https://github.com/google-research/google-research/tree/master/musiq,
    Ke et al., ICCV 2021). Non-commercial, coursework use only.
    The pretrained weights are the pyiqa release files, downloaded on demand.

Changes against pyiqa: attention uses F.scaled_dot_product_attention with a key
mask; the stem and embedding run only on valid patches; tokens are built on the
device from a centred canvas (see `build_tokens`); the native scale is capped
at `native_max_patches` by shrinking the image, not by cutting patches; the
resize is antialiased (`antialias=True`) unless asked otherwise.

Input: a canvas (B, C, S, S), C = 1 (grey, repeated to 3) or 3, valid pixels in
[0, 1], everything outside the photograph exactly -1. The photograph's box is
found from the sentinel.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 32
GRID = 10                        # hashed spatial position grid
SCALES = (224, 384)              # longer sides of the two resized scales

RELEASE = "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/"
WEIGHTS = {
    "koniq": ("musiq_koniq_ckpt-e95806b9.pth", "e95806b9"),
    "spaq": ("musiq_spaq_ckpt-358bb6af.pth", "358bb6af"),
}


# ------------------------------------------------------------------ padding

def exact_padding_2d(x, kernel, stride=1, dilation=1):
    """TensorFlow 'SAME' zero padding for a 4D tensor (pyiqa, constant mode)."""
    kernel = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    h, w = x.shape[-2:]
    h2, w2 = math.ceil(h / stride[0]), math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    return F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))


class ExactPadding2d(nn.Module):
    def __init__(self, kernel, stride=1):
        super().__init__()
        self.kernel, self.stride = kernel, stride

    def forward(self, x):
        return exact_padding_2d(x, self.kernel, self.stride)


# ------------------------------------------------------------------ network

class StdConv(nn.Conv2d):
    """Convolution with weight standardisation and 'SAME' padding."""

    def forward(self, x):
        x = exact_padding_2d(x, self.kernel_size, self.stride)
        w = self.weight
        w = w - w.mean((1, 2, 3), keepdim=True)
        w = w / (w.std((1, 2, 3), keepdim=True) + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        width = inplanes
        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=1e-4)
        self.relu = nn.ReLU(True)
        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=1e-4)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        return self.relu(x + identity)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """x (B, N, C); mask (B, N) bool, True for valid tokens (keys)."""
        B, N, C = x.shape
        h = self.num_heads
        q, k, v = (f(x).reshape(B, N, h, C // h).transpose(1, 2)
                   for f in (self.query, self.key, self.value))
        am = None if mask is None else mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.out(y.transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class AddHashSpatialPositionEmbs(nn.Module):
    def __init__(self, grid, dim):
        super().__init__()
        self.position_emb = nn.Parameter(torch.randn(1, grid * grid, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, x, pos):
        return x + self.position_emb.squeeze(0)[pos.long()]


class AddScaleEmbs(nn.Module):
    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, x, scale):
        return x + self.scale_emb[scale.long()]


class TransformerEncoder(nn.Module):
    def __init__(self, dim, mlp_dim=1152, num_heads=6, num_layers=14, num_scales=3, grid=GRID):
        super().__init__()
        self.posembed_input = AddHashSpatialPositionEmbs(grid, dim)
        self.scaleembed_input = AddScaleEmbs(num_scales, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder_norm = nn.LayerNorm(dim, eps=1e-6)
        self.transformer = nn.ModuleDict(
            {f"encoderblock_{i}": TransformerBlock(dim, mlp_dim, num_heads)
             for i in range(num_layers)})

    def forward(self, x, pos, scale, mask):
        x = self.scaleembed_input(self.posembed_input(x, pos), scale)
        x = torch.cat([self.cls.repeat(x.shape[0], 1, 1).to(x.dtype), x], dim=1)
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        for block in self.transformer.values():
            x = block(x, mask)
        return self.encoder_norm(x)


# ------------------------------------------------------------------ tokens

def native_size(h: int, w: int, cap: int, patch: int = PATCH) -> tuple[int, int]:
    """(h, w) itself when it has at most `cap` patches, else shrunk, aspect kept, to fit."""
    def count(a, b):
        return math.ceil(a / patch) * math.ceil(b / patch)
    if count(h, w) <= cap:
        return h, w
    s = math.sqrt(cap * patch * patch / (h * w))
    while True:
        rh, rw = max(1, round(h * s)), max(1, round(w * s))
        if count(rh, rw) <= cap:
            return rh, rw
        s *= 0.995


@lru_cache(maxsize=256)
def _hash_index(count_h: int, count_w: int, grid: int = GRID) -> torch.Tensor:
    """Hashed spatial position index of every patch of a count_h x count_w grid (row-major)."""
    g = torch.arange(grid).float()
    pw = F.interpolate(g.reshape(1, 1, grid), (count_w), mode="nearest").repeat(1, count_h, 1)
    ph = F.interpolate(g.reshape(1, 1, grid), (count_h), mode="nearest").transpose(1, 2)
    ph = ph.repeat(1, 1, count_w)
    return (ph * grid + pw).reshape(-1).long()


def patches_of(img: torch.Tensor):
    """(1, 3, h, w) in [-1, 1] -> patches (N, 3, 32, 32), hashed positions (N,), row-major."""
    x = exact_padding_2d(img, PATCH, PATCH)
    p = F.unfold(x, PATCH, stride=PATCH)                        # (1, 3*32*32, N)
    p = p[0].t().reshape(-1, 3, PATCH, PATCH)
    ch, cw = math.ceil(img.shape[-2] / PATCH), math.ceil(img.shape[-1] / PATCH)
    return p, _hash_index(ch, cw).to(img.device)


def _resize(img, rh, rw, antialias):
    if (rh, rw) == tuple(img.shape[-2:]):
        return img
    return F.interpolate(img, (rh, rw), mode="bicubic", align_corners=False, antialias=antialias)


def sample_scales(img: torch.Tensor, native_max_patches: int, antialias: bool = True):
    """One photograph (1, 3, h, w) in [-1, 1] -> [(patches, pos, scale_id)] for 224, 384, native."""
    h, w = img.shape[-2:]
    out = []
    for sid, longer in enumerate(SCALES):
        r = longer / max(h, w)
        p, pos = patches_of(_resize(img, round(h * r), round(w * r), antialias))
        out.append((p, pos, sid))
    rh, rw = native_size(h, w, native_max_patches)
    p, pos = patches_of(_resize(img, rh, rw, antialias))
    out.append((p, pos, len(SCALES)))
    return out


def image_box(canvas: torch.Tensor):
    """(top, bottom, left, right) of the photograph in a canvas (C, S, S): where it is >= 0."""
    valid = canvas[0] >= 0
    rows = valid.any(1).nonzero()
    cols = valid.any(0).nonzero()
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def build_tokens(canvas: torch.Tensor, native_max_patches: int = 768, antialias: bool = True):
    """Canvas batch -> flat valid patches and the padded token layout.

    Returns patches (V, 3, 32, 32); b_idx, slot (V,) where each valid patch goes in
    the (B, L) token grid; pos, scale, mask (B, L) with L the longest sample.
    """
    B = canvas.shape[0]
    per = []
    for b in range(B):
        top, bottom, left, right = image_box(canvas[b])
        img = canvas[b:b + 1, :, top:bottom, left:right]
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        img = (img.float() - 0.5) * 2
        parts = sample_scales(img, native_max_patches, antialias)
        per.append((torch.cat([p for p, _, _ in parts]),
                    torch.cat([q for _, q, _ in parts]),
                    torch.cat([torch.full_like(q, s) for _, q, s in parts])))
    L = max(p.shape[0] for p, _, _ in per)
    dev = canvas.device
    pos = torch.zeros(B, L, dtype=torch.long, device=dev)
    scale = torch.zeros(B, L, dtype=torch.long, device=dev)
    mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for b, (p, q, s) in enumerate(per):
        n = p.shape[0]
        pos[b, :n], scale[b, :n], mask[b, :n] = q, s, True
    patches = torch.cat([p for p, _, _ in per])
    b_idx, slot = mask.nonzero(as_tuple=True)
    return patches, b_idx, slot, pos, scale, mask


# ------------------------------------------------------------------ model

class Musiq(nn.Module):
    """MUSIQ on a centred canvas. `forward` is the raw quality score (0-100 scale) with the head."""

    def __init__(self, with_head: bool = True, native_max_patches: int = 768,
                 antialias: bool = True, dim: int = 384):
        super().__init__()
        self.native_max_patches, self.antialias = native_max_patches, antialias
        self.conv_root = StdConv(3, 64, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, 64, eps=1e-6)
        self.root_pool = nn.Sequential(nn.ReLU(True), ExactPadding2d(3, 2), nn.MaxPool2d(3, 2))
        self.block1 = Bottleneck(64, 256)
        self.embedding = nn.Linear(256 * (PATCH // 4) ** 2, dim)
        self.transformer_encoder = TransformerEncoder(dim)
        if with_head:
            self.head = nn.Linear(dim, 1)

    def features(self, canvas: torch.Tensor) -> torch.Tensor:
        patches, b_idx, slot, pos, scale, mask = build_tokens(
            canvas, self.native_max_patches, self.antialias)
        x = self.root_pool(self.gn_root(self.conv_root(patches)))
        x = self.block1(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        x = self.embedding(x)
        tokens = x.new_zeros(mask.shape[0], mask.shape[1], x.shape[-1])
        tokens[b_idx, slot] = x
        return self.transformer_encoder(tokens, pos, scale, mask)[:, 0]

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        f = self.features(canvas)
        return self.head(f).squeeze(1) if hasattr(self, "head") else f


# ------------------------------------------------------------------ weights

def weights_dir() -> Path:
    env = os.environ.get("MUSIQ_WEIGHTS_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/musiq")
    return Path.home() / ".cache" / "musiq"


def fetch_weights(name: str) -> Path:
    """Path of the checkpoint `name` ('koniq' or 'spaq'); downloaded once, hash checked."""
    fname, prefix = WEIGHTS[name]
    path = weights_dir() / fname
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        torch.hub.download_url_to_file(RELEASE + fname, str(tmp), hash_prefix=prefix)
        os.replace(tmp, path)
    return path


def load_musiq(weights: str = "koniq", with_head: bool = True, pretrained: bool = True,
               **kw) -> Musiq:
    model = Musiq(with_head=with_head, **kw)
    if not pretrained:
        return model
    state = torch.load(fetch_weights(weights), map_location="cpu", weights_only=True)
    if not with_head:
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
    model.load_state_dict(state, strict=True)
    return model
