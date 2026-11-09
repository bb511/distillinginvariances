#!/usr/bin/env bash
# ========================================================================
# MNIST CIRCULAR CNN HYPERPARAMETER SEARCH
# ========================================================================
# The teacher's Optuna study (02_teachers.md): TPE seeded at 1234 with 20
# start-up trials, 200 trials of 50 epochs on the main run of the canonical
# record, each maximising val/acc_best of the teacher stage. The space is the
# convolution channels, the kernel, the head, the learning rate and the weight
# decay (configs/hparams_search/teacher/mnist_cnn_optuna.yaml). The study lives
# in logs/optuna/teacher.db as cnn_circular_canonical; the trials draw no
# figure, log nowhere and write no results row. The zero-padded control is not
# searched: it takes the selected recipe through mnist_cnn_zeros.
#
# Hand-maintained. Every command is commented out; feed the file to
# scripts/cluster/submit_search.sh (olgpu) or submit_search_ngt.sh (a pod), and
# read the selected point off with
#   python3 scripts/search/studies.py best cnn_circular_canonical \
#       --db logs/optuna/teacher.db

# ----------------------------------------------------------------------
# 1. teacher  circular CNN  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=mnist_cnn \
#     hparams_search=teacher/mnist_cnn_optuna \
#     method=teacher \
#     data.regime=canonical \
#     hydra.sweeper.n_trials=200 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
