# MUSIQ Deliverable 2 report

## Files changed
- `src/data.py`: `build_photos(cfg, paths, targets, augment)` (MultiscalePhotos for `model.type: musiq`, otherwise `Photos` as before).
- `src/model.py`: `MusiqReg` + `build_model` branch `musiq`.
- `src/train.py`: `--model musiq`, `build_photos` in `loader()`, pixel statistics skipped when `model.multiscale`.
- `src/predict.py`, `scripts/sanity.py`: use `build_photos`. `predict` builds with `pretrained: false` via `load_checkpoint`, so it never downloads.
- `configs/musiq_koniq.yaml` (new), `README.md` (licence/weights line).

## Parameters
`MusiqReg`: 27,125,825 (backbone 27,125,440 + head 385). Frozen backbone: 385 trainable.

## Batch / accumulation
No measurement (user decision). Estimate: 0.6 GB per worst-case photograph in fp16 (962 tokens) + 0.5 GB state, so batch 8 is about 5.3 GB and batch 4 about 2.9 GB. Chosen batch 4 x accum 4 (effective 16), as instructed when unsure; batch 8 x accum 2 likely fits a 16 GB GPU too.

## Decisions
- Hyperparameters as specified: AdamW, finetune_lr 3e-5, head lr 2e-4, phases [head, all] [2, 8], huber 0.1, wd 0.05, warmup 0.15, fp16, early stopping 4, 2 workers, KonIQ weights, grey input.
- Existing models are untouched: `build_photos` gives the same `Photos` arguments; `max_shift` now defaults to 16 if missing from a config (previously a KeyError in train).
- `--epochs`, `--tta`, `--ensemble`, accumulation, phases rescaling are model-agnostic and unchanged.
- Licence note in the config header and README.

## Forward/backward
CPU, random-init `MusiqReg`, two 1024x1024 canvases (768x768 and 800x780 photos): output shape (2,), finite, all 248 parameter tensors received gradients. YAML parses; `import src.train, src.predict` OK.

## NOT tested
Pretrained weight download in the integrated path, any training run (head/all phases, fp16 scaler, early stopping, `--epochs`, `--fold all`), `--tta` validation, `src.predict` and `--ensemble` with a MUSIQ checkpoint, `scripts/sanity.py`, memory measurement, the SmallCNN/EffNet/ConvNeXt regression, the pytest suite, new unit tests for `build_photos`/`MusiqReg`.
