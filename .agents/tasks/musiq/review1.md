# Review 1: MUSIQ zero-shot pre-test (a9a62e7)

Verdict: APPROVED, no blockers.

Evidence
- Requirements met in `scripts/musiq_pretest.py`. It scores all training photos, which is 1841 when `--limit` is not set. It runs both koniq and spaq weights. It reports Spearman and Pearson of the raw score. It reports OOF RMSE from a linear mapping (headline) and an isotonic mapping, both fitted with `assign_folds` on `split.column` (group_strict, 4 folds), each fold predicted from a fit on the other three.
- The cell text is in report1.md and in the script docstring. It uses /kaggle/working/repo and the `data/work/reference/train_1024` cache. The TRAIN_IMG, TRAIN_CSV and COMP paths belong to the setup cell, which links data into `data/raw` as README.md lines 314-317 describe. The plan states the setup cell does this. The script reads data only through the repo config, so those paths resolve.
- Recorded verification: pytest gives 11 passed, 1 skipped. A 120-photo MPS smoke run produced plausible numbers for all four weights/input combinations. The full run was not done locally and is expected on Kaggle. That is acceptable for this deliverable.
- Scope: the commit touches only scripts/musiq_pretest.py, src/data.py (additive `MultiscalePhotos`), src/musiq.py and tests/test_musiq.py. Existing `Photos` behaviour is unchanged.
- User-owned state: the two configs/_tmp_lr*.yaml deletions are still unstaged, and `.agents/` is untracked and not in the commit (`git diff --name-only` lists only the four files above).
- Push: `origin/main` is at 0879d17 and HEAD is one commit ahead. `upstream/main` is unchanged at b99bf92. Nothing was pushed.
