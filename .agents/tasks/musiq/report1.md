# Report 1: MUSIQ zero-shot pre-test

Commit: a9a62e7 (local, not pushed). Files: `src/musiq.py`, `src/data.py` (`MultiscalePhotos`), `scripts/musiq_pretest.py`, `tests/test_musiq.py`.

## Kaggle cell
```python
import os
os.chdir("/kaggle/working/repo")
os.environ.update(PYTHONPATH="/kaggle/working/repo", WANDB_MODE="disabled",
                  MUSIQ_WEIGHTS_DIR="/kaggle/temp/musiq")
assert os.path.isdir("data/work/reference/train_1024"), "run python -m src.prepare first"
!python scripts/musiq_pretest.py --weights koniq spaq --input both --batch-size 8 --workers 2
```
Needs Internet on (two weight downloads, ~216 MB). Prints Spearman, Pearson and OOF RMSE per weights/input and writes CSVs to `models/musiq_pretest`.

## Local results (MPS, 120 photos spread evenly, 116 eval_ok; full set ~50 min locally, not run)
| weights | input | Spearman | Pearson | OOF RMSE linear (eval_ok) | isotonic (eval_ok) |
|---|---|---|---|---|---|
| koniq | rgb | 0.166 | 0.291 | 0.2306 | 0.2356 |
| koniq | grey | 0.188 | 0.293 | 0.2327 | 0.2478 |
| spaq | rgb | 0.261 | 0.202 | 0.2378 | 0.2390 |
| spaq | grey | 0.268 | 0.253 | 0.2388 | 0.2415 |

The subset is too small to compare with 0.1956 / ~0.145; the Kaggle run on all 1841 photos gives those numbers. Tests: `pytest tests -q` gives 11 passed, 1 skipped (network test).

## Mapping
Headline is a 2-parameter linear fit clipped to [0, 1], fitted on three group_strict folds and applied to the fourth. It cannot overfit and does not depend on the MOS scale. Isotonic regression is the second column (the best any monotone mapping can do). RMSE is pooled out-of-fold, on eval_ok rows and on all rows.

## Weights
Vendored MUSIQ (no pyiqa dependency), weights from the pyiqa GitHub release with sha256 prefix check. Cache: `$MUSIQ_WEIGHTS_DIR`, else `/kaggle/temp/musiq` if it exists, else `~/.cache/musiq`. On Kaggle the cell sets `/kaggle/temp/musiq`, so the files stay out of the notebook output.
