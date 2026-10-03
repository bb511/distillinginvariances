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
The jets are prepared by the loader that ships with the record.
It orders the constituents of each jet by transverse momentum, keeps the leading 50 and pads jets with fewer constituents with zeros; no cut is placed on the transverse momentum of a constituent.
Each of the 16 columns is then divided by a scale fitted on the record's training split, which leaves the padding at zero.
The four columns measured from the jet axis (`etarel`, `phirel`, `deltaR` and `ptrel`) are recomputed from the 50 constituents kept, since the record measures them from the axis of the untruncated jet.
The digits are normalised by the mean and standard deviation of the MNIST training pixels.

The training data of each record form a pool: the training and validation splits of the jets (620k jets) and the training split of MNIST (60k images).
The pool is cut into five folds, each validating on one fifth of the pool and training on the other four; the test splits (260k jets and 10k images) are never pooled.
The main run trains and validates on the record's own split instead (496k and 124k jets); MNIST has no validation split, so its main run holds out a seeded 10k of the training images.
The main run serves the hyperparameter searches and the quantised student; it enters none of the tables averaged over folds.

Every run is made in one of three regimes.
In the canonical regime, the data are served as the record stores them.
In the transformed regime, every pool and test row is replaced once, before the pool is folded, by a random draw from the teacher's group.
In the augmented regime, the training rows are transformed afresh in every batch; validation and testing use the rows of the transformed regime, and the teacher is the canonical one.

## Training

An experiment config fixes the data set and the teacher: `mnist_cnn`, `jets_deepsets` or `jets_lorentznet`.
The stage is chosen with `method`, the regime with `data.regime` and the fold with `fold`:

```bash
python src/train.py experiment=jets_lorentznet method=teacher data.regime=transformed fold=1
python src/cache.py experiment=jets_lorentznet data.regime=transformed fold=1
python src/train.py experiment=jets_lorentznet method=kd data.regime=transformed fold=1
```

Leaving `fold` unset selects the main run, which the hyperparameter searches use:

```bash
python src/train.py experiment=jets_lorentznet method=teacher
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
