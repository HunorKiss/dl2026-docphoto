# Review 2: MUSIQ integration (static)
Verdict: APPROVED. No blockers.

Checked (diff vs HEAD, plus src/optim.py, src/evaluate.py, src/data.py, src/musiq.py load path):
- train.py: `--model musiq` is accepted. `build_photos` feeds MultiscalePhotos for musiq and Photos with the same arguments otherwise. `pixel_stats` is skipped only when `model.multiscale` is set, which only MusiqReg does, so the other three models behave as before.
- Phases and learning rates: MusiqReg provides `head_parameters` and `set_backbone_frozen`. The "all" phase builds param_groups with the head group last (backbone 3e-5, head 2e-4). `lr_head` is logged via `lrs[-1]`. `--epochs` rescales `phase_epochs`, and nothing in it depends on the model.
- fp16 scaler, accumulation, clipping and huber all sit in the model-agnostic loop. MusiqReg returns a logit of shape (B,), the same as the other models.
- TTA: `x.flip(-1)` on a centred canvas keeps the -1 sentinel, so the photograph box is still found. The canvas shifts by at most 1 px, which is harmless.
- Checkpoint: `load_checkpoint` sets `pretrained: false`, so `load_musiq` returns before `fetch_weights` and predict never downloads. The state_dict holds the backbone, head and in_mean/in_std buffers, so loading is strict-compatible. predict builds its dataset through `build_photos(cfg, paths)` and `--ensemble` uses the same path.
- User state: the deleted `configs/_tmp_lr10e4.yaml` and `configs/_tmp_lr10e5.yaml` are still deleted. `.agents/` is untracked and was not touched. No temp files appear in `git status`.

Not verified (static review only): a real training run, the pretrained download, and memory use. report2.md states these as untested. They are not blockers.
