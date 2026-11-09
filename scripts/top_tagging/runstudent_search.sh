#!/usr/bin/env bash
# ========================================================================
# TOP-TAGGING STUDENT HYPERPARAMETER SEARCH
# ========================================================================
# One Optuna study per block (03_student.md): TPE seeded at 1234 with 20 start-up
# trials, 200 trials of 50 epochs on the main run of the canonical record, each
# maximising val/acc_best of the cross-entropy baseline. A stack above 30k
# multiply-accumulates scores 0 without loading data and counts among the 200.
# The studies live in logs/optuna/student.db under the experiment's search_study
# and the regime; the trials draw no figure, log nowhere and write no results row.
#
# Hand-maintained. Every command is commented out; feed the file to
# scripts/cluster/submit_search.sh (olgpu) or submit_search_ngt.sh (a pod), and
# read the selected point off with scripts/search/studies.py.

# ----------------------------------------------------------------------
# 1. student  [pt, eta, phi] of 100 constituents  (LorentzNet cell)
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=student/top_tagging_pt_eta_phi \
#     hparams_search=student/jets_optuna \
#     method=baseline \
#     data.regime=canonical \
#     hydra.sweeper.n_trials=200 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
