#!/usr/bin/env bash
# ========================================================================
# MNIST, TEACHER-FREE STUDENT ROWS: TRAINING COMMANDS
# ========================================================================
# The rows no teacher enters: the baseline, the label-smoothed students (epsilon 0.05, 0.1,
# 0.2, a reported curve, never selected) and the augmented baseline (the ceiling of Block 2's
# margin and of the transfer fraction), under student/mnist, whose teacher is none, so that
# they train once in mnist/none/... and the MNIST cell evaluates them (the TEACHER-FREE ROWS
# of runcnn_eval.sh). Canonical once, transformed and augmented under translation. The
# student's learning rate and weight decay are those of student/mnist, filled in once
# Student I has selected the student.
#
# Every block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition), independent of every other: these rows can run at any time.
# Folds and extra seeds are submission parameters, and --only takes block indices:
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 scripts/mnist/runstudent_rows_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 scripts/mnist/runstudent_rows_train.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 --folds 1-5 scripts/mnist/runstudent_rows_train.sh \
#       callbacks=default
# [seeds]: the headline rows, baseline and augmented, also submitted with a trailing
# seed=12346 and seed=12347, on the main run only. [folds]: every block, with --folds 1-5 and
# a trailing callbacks=default.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# 1. baseline  canonical  [seeds] [folds]
# python3 src/train.py experiment=student/mnist method=baseline data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. baseline  transformed translation  [seeds] [folds]
# python3 src/train.py experiment=student/mnist method=baseline data.regime=transformed \
#     data.group=translation logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 3. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/mnist method=label_smoothing data.regime=canonical \
#     algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 4. label_smoothing  transformed translation  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/mnist method=label_smoothing data.regime=transformed \
#     data.group=translation algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 5. baseline  augmented translation  [seeds] [folds]
# python3 src/train.py experiment=student/mnist method=baseline data.regime=augmented \
#     data.group=translation logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
