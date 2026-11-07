# Experiment log

It is already at the root of your repository; fill it in as you go.

**What counts as a step.** One change to the data handling, the network, the
training or the prediction, applied on top of the previous step, with the
reason you expected it to help. Another seed, a rerun or the same change twice
is not a step. An ensemble counts once, however many models are in it.

**What counts as an improvement.** The out-of-fold RMSE of the four
`group_strict` folds (the four `*_val.csv` files pooled, `eval_ok` rows) is
lower than the step before. The leaderboard does not count here.

The signature needs at least four steps that improved, in one chain from the
reference to your selected submission. Steps that did not help stay in the
table, marked "no", and the next step builds on the last one that did.

| # | change | why I expected it to help | config | fold 0 | fold 1 | fold 2 | fold 3 | out-of-fold | kept? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | the reference, your own run | | `configs/reference.yaml` | 0.1769 | 0.1897 | 0.2316 | 0.1772 | 0.1956 | yes |
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 3 | | | | | | | | | |
| 4 | | | | | | | | | |

**Selected submission:** step __, `submissions/<file>.csv`, public RMSE ____,
checkpoint `<path or link>`.
