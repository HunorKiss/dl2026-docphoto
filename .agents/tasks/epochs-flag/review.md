# Review: --epochs flag (aa0b96e...HEAD)

Changed files: src/train.py only (+35). git status shows only the user's two uncommitted deletions (configs/_tmp_lr10e4.yaml, configs/_tmp_lr10e5.yaml); no temp files.

Evidence:
- Override applied at train.py:139-145, after cfg load and --seed/--model, before run_name (line 151) and tracking.start (155), so W&B config carries the new train.epochs and the run name/output files are unchanged. Without the flag the block is skipped: behavior identical.
- Schedule: make_opt_sched builds per-phase totals from phase epochs (line 288), so warmup fraction and cosine are recomputed. Plain configs use t["epochs"] (233/244).
- Rescale (77-98): checked by hand [4,8]->20 gives [7,13]; floors+min 1, then add/remove loops terminate and sum to N because N >= k is enforced; N<k raises ValueError with a clear message; N<1 also raises. Configs without phase_epochs keep the even split (242-245). Rounding rule is documented in the docstring and commit message.
- Effective epochs printed via the "training phases" line and the rescale message; docstring usage line added.
- Verification recorded in commit 0879d17: no-flag run unchanged, yaml epochs 3 equals yaml 1 + --epochs 3, rescale cases and N=1 error, --fold all run. Sufficient.

Non-blocking note (not filed): legacy freeze_epochs path with --epochs <= freeze_epochs is not guarded; no config uses freeze_epochs and the task only specifies phase_epochs.

Verdict: APPROVED, no blockers.
