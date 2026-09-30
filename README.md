# Distilling symmetries

Can a plain multilayer perceptron learn a symmetry from a teacher that has it built in?
Here, three symmetric teachers are distilled into perceptron students, and the share of the symmetry each student acquires is measured.
The teachers are a circularly padded CNN, invariant under translations of MNIST digits; a Deep Sets network, invariant under permutations of the constituents of a jet; and LorentzNet, invariant under Lorentz boosts of the same jets.
Each teacher is distilled by logit matching, by a hint on one hidden layer and by relational distillation, both on the original data and on data transformed once by the group.

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/):

```bash
poetry install --no-root
bash scripts/setup.sh
```

The second command creates the working directories and the `.env` file that `configs/paths/default.yaml` reads.
`HF_HUB_CACHE` in `.env` sets where the data sets are downloaded.

## Data

Both data sets are read from HuggingFace at a pinned revision: the hls4ml LHC jet data set from [`fastmachinelearning/hls4ml_lhc_jets_150p`](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p) (Zenodo record 3602260) and MNIST from [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist).
The training data are cut into five folds; the test splits are kept whole.

## Training

An experiment config fixes the data set and the teacher: `mnist_cnn`, `jets_deepsets` or `jets_lorentznet`.
The stage is chosen with `method`, the regime with `data.regime` and the fold with `fold`:

```bash
python src/train.py experiment=jets_lorentznet method=teacher data.regime=transformed fold=1
python src/cache.py experiment=jets_lorentznet data.regime=transformed fold=1
python src/train.py experiment=jets_lorentznet method=kd data.regime=transformed fold=1
```

The teacher is trained first and its outputs are cached; every student is then distilled from that cache.
The commands that reproduce the study are listed, in the order they run, in `scripts/`.

## Evaluation

```bash
python src/eval.py experiment=jets_lorentznet method=kd data.regime=transformed fold=1
```

Each evaluation appends two rows to `results/results.csv`, one for the validation fold and one for the test split, with the accuracy, calibration, fidelity to the teacher and invariance of the model.

## Licence

The code is under the Apache License 2.0; see [`LICENSE`](LICENSE).
The data sets are distributed separately under their own licences.
