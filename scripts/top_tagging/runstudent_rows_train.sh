#!/usr/bin/env bash
# ========================================================================
# TOP TAGGING, TEACHER-FREE STUDENT ROWS: TRAINING COMMANDS
# ========================================================================
# The rows no teacher enters: the baseline, the label-smoothed students (epsilon 0.05, 0.1,
# 0.2, a reported curve, never selected) and the augmented baseline (the ceiling of Block 2's
# margin and of the transfer fraction), under student/top_tagging_pt_eta_phi, whose teacher is
# none, so that they train once in top_tagging/none/... and the LorentzNet cell evaluates them
# (the TEACHER-FREE ROWS of runlorentznet_m2_eval.sh). The blocks mirror the [pt, eta, phi]
# rows of scripts/jets/runstudent_rows_train.sh at the cell's group alone, lorentz_iso,
# without the double-width and four-momentum variants. The student's learning rate and weight
# decay are those of the selected student, recorded in the comment of configs/net/mlp_top.yaml
# (0.000707422426877928, 1e-07). Top tagging has a single student input, so no hp_tag carries
# a prefix. Every transformed and augmented block names its group, which sets the regime_tag
# of its run directory.
#
# Block 1 at seed 12345 on the main run trained on 9 Nov 2026
# (logs/train/top_tagging/none/canonical/baseline/main/lr0.000707422426877928_s12345), the
# teacher-free canonical baseline at seed 12345: on the main run block 1 is submitted with the
# extra seeds alone. A second submission at seed 12345 returns at that run's manifest when
# the config_hash agrees, and is refused otherwise.
#
# Every block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition), independent of every other: these rows can run at any time.
# Folds and extra seeds are submission parameters, and --only takes block indices, since a
# label (<experiment>_<regime>_<method>_main) does not name the group:
#   bash scripts/cluster/submit_ngt.sh --only 2,5 scripts/top_tagging/runstudent_rows_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 scripts/top_tagging/runstudent_rows_train.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 --folds 1-5 scripts/top_tagging/runstudent_rows_train.sh \
#       callbacks=default
# [seeds]: the headline rows, baseline and augmented, also submitted with a trailing
# seed=12346 and seed=12347, on the main run only. [folds]: the same headline rows (08_tests.md,
# Block 8, 10 Nov 2026), with --folds 1-5 and a trailing callbacks=default; the label-smoothing
# curves stay on the main run.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# [pt, eta, phi]
# ======================================================================

# 1. baseline  canonical  [seeds] [folds]
# python3 src/train.py experiment=student/top_tagging_pt_eta_phi method=baseline algorithm.optimizer.weight_decay=1e-07 algorithm.optimizer.lr=0.000707422426877928 data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. baseline  transformed lorentz_iso  [seeds] [folds]
# python3 src/train.py experiment=student/top_tagging_pt_eta_phi method=baseline algorithm.optimizer.weight_decay=1e-07 algorithm.optimizer.lr=0.000707422426877928 data.regime=transformed \
#     data.group=lorentz_iso logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 3. label_smoothing  canonical  epsilon curve
# python3 src/train.py -m experiment=student/top_tagging_pt_eta_phi method=label_smoothing algorithm.optimizer.weight_decay=1e-07 algorithm.optimizer.lr=0.000707422426877928 data.regime=canonical \
#     algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 4. label_smoothing  transformed lorentz_iso  epsilon curve
# python3 src/train.py -m experiment=student/top_tagging_pt_eta_phi method=label_smoothing algorithm.optimizer.weight_decay=1e-07 algorithm.optimizer.lr=0.000707422426877928 data.regime=transformed \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 5. baseline  augmented lorentz_iso  [seeds] [folds]
# python3 src/train.py experiment=student/top_tagging_pt_eta_phi method=baseline algorithm.optimizer.weight_decay=1e-07 algorithm.optimizer.lr=0.000707422426877928 data.regime=augmented \
#     data.group=lorentz_iso logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
