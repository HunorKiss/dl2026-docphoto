# data

Nothing in here is tracked by git. The directory layout is, so that everyone's
paths match. You download the files of `raw/` yourself (`../README.md`,
"The data"):

```
raw/train.csv                   the labels and the grouping columns    } the training dataset
raw/train/                      1841 training photographs (.jpg)       } released earlier
raw/test/                       1628 test photographs                  } the competition's
raw/sample_submission.csv       the image_id of every test photograph  } Data tab
work/reference/                 resized copies, written by python -m src.prepare
```

Treat `raw/` as read only. If a script needs a different form of the data it
writes it to `work/`, so that deleting `work/` and regenerating it is always
safe.
