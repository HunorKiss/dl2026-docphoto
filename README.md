# The reference solution

This is the model behind the **Reference solution** line on the leaderboard of
the **Snap or Retake: Document Photo Usability** competition, and a repository
you can start the competition from. It is lab 4's network with a setting from
lab 4 that works well enough, trained on all the training photographs: a small
convolutional network of 25 thousand weights, trained from scratch on grey
768 × 768 px pages. About four hundred lines of Python.

**Using it is a suggestion, not a requirement.** You may build on your lab
repositories, on the course template or on anything else you write. Whatever
you build on, the reference is the starting point you are measured from: the
signature line below is defined by its score, and the first row of your
experiment log is the reference, scored on the same four folds as everything
after it (`assign_folds` in `src/data.py`).

For the signature you improve on the reference in at least four validated
steps, and your selected submission must reach a private RMSE at least 25 %
below the reference's, that is **0.157 or lower** (`practices/README.md`,
section 6).

| RMSE | 4-fold CV by `group_strict` | public leaderboard | private leaderboard |
|---|---|---|---|
| constant (the training mean, 0.748) | 0.230 | 0.280 | 0.275 |
| **this reference** | **0.194** | **0.216** | **0.210** |
| the signature line | | | 0.157 |

**The leaderboard numbers are higher than the cross-validation ones, for every
model.** The test pages are new layouts photographed with other phones in other
rooms, and their labels are more spread out: even the constant does worse
there. Compare your models with each other on the folds; the leaderboard tells
you where you stand, not which of two close variants is better.

One run takes about 2 minutes on an NVIDIA RTX 3500 Ada laptop GPU and about
20 minutes on **four laptop CPU cores**, so the reference itself needs no GPU.
What you build on it probably will (see [below](#without-a-gpu-of-your-own)).

## Set it up

This folder is lab 3's Docker environment plus the reference code. How you
keep it in git is up to you: a new repository of its own (the simplest, and
what the commands below do), or a separate branch of a repository you already
have. Whichever you choose, the repository you hand in must have **only your
solution on its `main` branch** ([What you hand in](#what-you-hand-in)).

```bash
git -C <course> pull                                           # Linux, macOS, Git Bash
cp -r <course>/reference_solution ~/snap-or-retake
cd ~/snap-or-retake
git init && git add . && git commit -m "reference solution"
```

```powershell
git -C <course> pull                                           # Windows PowerShell
Copy-Item -Recurse <course>\reference_solution $HOME\snap-or-retake
cd $HOME\snap-or-retake
git init; git add .; git commit -m "reference solution"
```

Commit it as it is before you change anything, so your own changes show up as
diffs against the reference. Push it to a **private** repository: rule 7 asks
for the code behind your submission.

**Docker**, as in lab 3:

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
docker compose build          # instant if lab 3's image is still there: same name, same content
docker compose run --rm dev python scripts/check_env.py
```

**On a machine with an NVIDIA GPU** the GPU has to be switched on, in one of
two ways (pick one):

* **once, in `.env`** (recommended): uncomment the `COMPOSE_FILE` line for your
  operating system. Every command in this README then uses the GPU as written;
* **or in every command**: leave `.env` alone and put
  `-f docker-compose.yml -f docker-compose.gpu.yml` after `docker compose`, for
  example
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev python scripts/check_env.py`.

Either way, `check_env.py` then prints a `gpu` line with the card's name; if it
says `not available`, the run is on the CPU.

**Weights & Biases**, as in labs 3 and 4: every training run logs to it
([below](#weights--biases)). Paste your key from https://wandb.ai/authorize
into `.env` after `WANDB_API_KEY=` now; your lab-3 `.env` has it already.

## The data

Two downloads, from two places. `data/raw/` in this repository is empty
(only a `.gitkeep`): you put the files there yourself, and nothing in `data/`
goes into git.

| what | where it comes from |
|---|---|
| `train.csv`, `train/` (1841 .jpg) | **the training dataset released earlier**, the one you used in the labs. Copy both from your lab repository's `data/raw/`, or download the dataset again: its link is in the competition's *Data* description |
| `test/` (1628 .jpg), `sample_submission.csv` | **the competition**: its *Data* tab, *Download All* |

Unpack them so that exactly these four exist:

```
data/raw/train.csv                the labels and the grouping columns
data/raw/train/                   1841 training photographs
data/raw/test/                    1628 test photographs
data/raw/sample_submission.csv    the image_id of every test photograph
```

## Run it

Three commands, from the repository root:

```bash
docker compose run --rm dev python -m src.prepare                   # resize once, ~3 min
docker compose run --rm dev python -m src.train --fold all          # train on every photograph
docker compose run --rm dev python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
```

The last one writes `submissions/reference.csv`. Check it before you upload
it:

```bash
docker compose run --rm dev python scripts/check_submission.py submissions/reference.csv
```

Its score will be close to the reference line but probably not equal to it:
another seed moves the private score anywhere between 0.204 and 0.216, and
another GPU or a CPU does the arithmetic in a slightly different order.

## Validate on the folds

**Why not just use the leaderboard.** Kaggle scores every upload on all test
photographs but shows you only part of the result. The **public leaderboard**
is the RMSE on about a third of the test pages, shown right after each upload.
The **private leaderboard** is the RMSE on the other two thirds; it stays
hidden until the deadline, and it alone decides the ranking and the signature
line. A third of the pages is a handful of sheets, so two versions of a model
that differ by a few thousandths on the public board are often in the opposite
order on the private one. Worse, every choice you make by looking at the public
score fits those few pages a little more: after a few weeks of tuning on it,
the public score flatters you and the private one, revealed at the end, does
not. With five uploads a day you also cannot test much there. The folds have
none of these problems: they are yours, as many runs as you like, and the
photographs they hold out never shaped the model.

Train on three folds and score the fourth:

```bash
docker compose run --rm dev python -m src.train --fold 0             # folds 1-3 train, fold 0 validates
```

It prints the validation RMSE after every epoch and writes the predictions to
`models/reference/*_val.csv`. Run all four (`--fold 0`, `1`, `2`, `3`); the
number to compare is the RMSE of the four validation files pooled together,
the out-of-fold RMSE, which is 0.194 for the reference. The fold split is by
`group_strict` (lab 3): the six strict units of the training set are dealt
into 4 folds of 370 to 560 photographs, and no page or layout is on both
sides.

Everything that changes the result is in `configs/reference.yaml`. To try
something, copy it (`cp configs/reference.yaml configs/my_try.yaml`), change
the copy, and pass `--config configs/my_try.yaml` to `train`. Keep every
config you submitted from; rule 7 asks you to reproduce the submission.

## Before a long run: the checks

Whenever you change the network, the loss or the input, run lab 4's checks
before you spend an hour of training on the change:

```bash
docker compose run --rm dev python scripts/sanity.py --config configs/my_try.yaml
```

They take seconds on a GPU and a minute or two on a CPU, and each prints PASS
or FAIL:

1. **one forward pass**: one finite number per photograph, a prediction in [0, 1];
2. **the loss before training**: about the loss of predicting the training
   mean, because the output bias starts there;
3. **overfit one batch**: trained on 16 photographs only, the loss falls below
   0.001. If it cannot, there is a bug, and more data will not fix it;
4. **the same seed twice**: two short runs from one seed give the same losses,
   or no two of your runs can be compared.

The checks use the network (`src/model.py`), the image size of the config, and
the loss with its prediction (`src/loss.py`), the same ones `train.py` and
`predict.py` use. A failed check exits with an error, so
`python scripts/sanity.py --config ... && python -m src.train ...` never starts
a run that cannot learn.

## Weights & Biases

Every run of `src.train` logs its config, its curves (training and validation
RMSE, learning rate, per epoch) and its final validation RMSE to the W&B
project `snap-or-retake`. The four fold runs of one experiment share a group
named after their `--out` folder, so `models/lr1e-3` shows up as the group
`lr1e-3`. `EXPERIMENTS.md` stays the record you hand in; W&B is where you
compare curves and look up what a run did.

`WANDB_MODE` in `.env` decides where the logs go: `online` (the default),
`offline` (into `wandb/`, uploaded later with `wandb sync wandb/offline-run-*`),
or `disabled`. Without a key the run is logged anonymously, with a link you can
claim later; without a network it falls back to offline. Logging never changes
a result: the run starts before the seed is set.

## What is in it, and why

| file | what |
|---|---|
| `configs/reference.yaml` | data paths, image size, the network's widths, the training schedule |
| `src/config.py` | reads the config; the seed, the device, mixed precision |
| `src/data.py` | the tables, the folds, the resize, the letterbox, the augmentation: everything that touches pixels |
| `src/model.py` | the network, and loading it back from a checkpoint |
| `src/loss.py` | the loss, and how the network's output becomes a prediction: change the two together |
| `src/optim.py` | the parameter groups and the learning-rate schedule |
| `src/evaluate.py` | predictions for a set of photographs, and the RMSE |
| `src/prepare.py` | resizes every photograph once to 1024 px on the long side |
| `src/train.py` | one training run, on three folds or on all of them: puts the pieces above together |
| `src/predict.py` | one or more checkpoints in, a submission out |
| `src/tracking.py` | the Weights & Biases logging of `train.py` |
| `scripts/sanity.py` | the checks before training (lab 4) |
| `scripts/check_submission.py` | checks a submission file before you upload it |
| `scripts/check_env.py` | checks the container, as in lab 3 |
| `EXPERIMENTS.md` | the template of your experiment log |
| `Dockerfile`, `docker-compose*.yml`, `requirements.txt` | lab 3's environment, unchanged |

The choices, and the lab each one comes from:

* **Validation by `group_strict`** (lab 3). A random split would put
  photographs of the same page on both sides and flatter the score by a fifth.
* **Grey pages at 768 px through a 7 × 7 stride-4 stem** (lab 4). The label is
  mostly about fine blur; at 256 px it is gone. The stem reads every pixel
  and hands a 192 px map to the rest of the network. The resize uses area
  averaging for the same reason.
* **Conv-BatchNorm-ReLU blocks, average and maximum pooling side by side**
  (lab 4). The average says how sharp the page is overall, the maximum how
  bad its worst region is.
* **He initialisation, the output bias at the training mean** (lecture 4,
  lab 4). The network returns a logit; the sigmoid is applied in the loss and
  in `predict`.
* **AdamW with weight decay on the weights only, warm-up then cosine, gradient
  clipping, 12 epochs, the last epoch kept** (lecture 5, lab 4). The number of
  epochs is fixed in advance, so the `all` run needs no validation set and a
  fold run measures exactly what the `all` run does.
* **Mirror and shift only** (lab 4). Brightness and blur are what the label
  measures; augmenting them trains the network on wrong labels.
* **MSE loss**, because the metric is RMSE.

## Where to go from here

The reference is where lab 4 ended. None of the techniques that came after it
in the lectures and the labs is in it, and much of that matters on this data. Which parts,
and how much, is what your experiments are for. Code from your lab
repositories may be brought over, as long as it ends up as part of your
solution.

Two habits make the difference between four steps and four guesses:

* **One change at a time, four folds each.** A change that does not lower the
  out-of-fold RMSE does not help, however reasonable it sounds. Small
  differences on a single fold are noise.
* **Look at the errors.** The `*_val.csv` files have `family`. Where is the
  model worst, and why?

## The experiment log

The signature asks for at least four changes, each of which lowered the
out-of-fold RMSE compared with the step before it. Record them in
`EXPERIMENTS.md` at the root of this repository, as you go, not on the
deadline. Changes that did not help belong in it too; they just do not count
towards the four.

## What you hand in

The full rules are in `practices/README.md`, section 5. In short, four things:

1. **Kaggle predictions**, uploaded to the competition, with your final
   submission selected on *My Submissions*.
2. **A git repository whose `main` branch is your solution and nothing else**,
   runnable by someone else from its README:
   * the code that produced the selected submission, and the commands that
     run it;
   * the config of every step in your experiment log, so that each number in
     it can be reproduced;
   * `EXPERIMENTS.md` (item 4) and a README.

   Not on `main`: lab code, notebooks of abandoned ideas, copies of other
   pipelines, anything the submission does not use. Keep those on another
   branch or in another repository. A commit history that shows the work is
   welcome on `main`; dead code in its final state is not.
3. **The trained checkpoint behind your selected submission**: committed if it
   is small enough, otherwise a download link in the README with the file's
   size and SHA-256 checksum.
4. **The experiment log**, `EXPERIMENTS.md`: one table, from the reference to
   your selected submission.

## Without a GPU of your own

**Kaggle Notebooks** give a free weekly GPU quota (*Settings → Accelerator →
GPU*; your phone number has to be verified). Kaggle sets the number of hours
itself: it is 30 a week as this is written, shown under the accelerator in
*Session options*. In a new notebook, add the competition and
the training dataset as inputs (*Add Input*), switch *Internet* on (for
`git clone`, W&B and any pretrained weights), add your W&B key as a Kaggle
*Secret* named `WANDB_API_KEY`, then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")   # or WANDB_MODE=disabled
os.environ["PYTHONPATH"] = "."          # what the Docker image sets: scripts/sanity.py imports src
!git clone https://<token>@github.com/<you>/snap-or-retake.git repo
%cd repo
!ln -s /kaggle/input/<training-dataset>/train data/raw/train
!cp /kaggle/input/<training-dataset>/train.csv data/raw/
!ln -s /kaggle/input/<competition>/test data/raw/test
!cp /kaggle/input/<competition>/sample_submission.csv data/raw/
!python -m src.prepare && python -m src.train --fold all
!python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
```

Look up the two `/kaggle/input/...` folder names in the notebook's input panel.
A private repository needs a GitHub token with read access in the URL (add it
as a Kaggle *Secret*, never in the notebook's text). Download the checkpoint
and the submission from the *Output* panel and commit the config you trained
with; the checkpoint goes where [What you hand in](#what-you-hand-in) says.
**Colab** works the same way with the data copied to a private Google Drive.
Either way the photographs stay private: never a public notebook, never a
public link.
