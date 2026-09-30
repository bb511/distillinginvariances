# Scripts

The commands behind every result in this study.
**Run everything from the repository root.**

- `mnist/`, `jets/`: the command catalogues of each experiment, `run<teacher>_train.sh` and `run<teacher>_eval.sh`.
- `cluster/`: `submit.sh`, which runs a catalogue on a GPU machine without a scheduler; `submit_ngt.sh`, its counterpart inside an NGT pod; and `setup_pod.sh`, which prepares the checkout in a new pod.
- `analysis/`: hyperparameter selection, averages over folds, paired differences and compute costs, all read from `results/results.csv`.

## Command catalogues

A catalogue is a list of commented-out `src/train.py`, `src/cache.py` and `src/eval.py` blocks, in the order they must run: the teacher, its cache and then the students, all on fold 1.
Running a file does nothing; the submitters parse it and queue the blocks you select:

```bash
bash scripts/cluster/submit.sh --list scripts/jets/rundeepsets_train.sh
bash scripts/cluster/submit.sh --only kd --gpu 1 scripts/jets/rundeepsets_train.sh
```

Once `analysis/select_hparams.py` has picked each method's hyperparameters on fold 1, `--folds 2-5` runs the selected blocks on the remaining folds.
