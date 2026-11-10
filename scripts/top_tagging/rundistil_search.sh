#!/usr/bin/env bash
# ========================================================================
# TOP TAGGING, DISTILLATION HYPERPARAMETER SEARCHES
# ========================================================================
# One Optuna study per block, the sibling's protocol (Block 3 of 08_tests.md): TPE seeded at
# 1234 with 20 start-up trials, 50 epochs on the main run, each trial maximising
# val/acc_best, its study row in logs/optuna/distillation.db and nothing else kept (numbered
# trial directories under logs/searches/multiruns/, no logger, no results row). Per regime:
#   kd, kd_hint, logit_matching   the learning rate, 20 trials, at T = 4 and alpha 0
#                                 (configs/hparams_search/distillation/lr_optuna.yaml;
#                                 logit_matching does not read T); with 20 start-up trials
#                                 every trial is a seeded random draw
#   relational, per alpha         learning rate, lambda_IS and lambda_IC, 60 trials, cosine
#                                 (relational_optuna.yaml); the warm-up cosine schedule
#                                 spans the 50 epochs, its length following trainer.max_epochs
# Every trial keeps the selected student's weight decay, 1e-07, as the comment of
# configs/net/mlp_top.yaml records it. The blocks mirror the LorentzNet-m2 blocks (11-20) of
# scripts/jets/rundistil_search.sh.
#
# The study is <dataset>_<teacher>_<regime_tag>_<stage>, or
# <dataset>_<teacher>_<regime_tag>_relational_a<alpha>, set by a study_name override on every
# block: the search configs name a study <teacher>_<regime_tag>_<stage>, which for this
# teacher, lorentznet_m2 like the jets cell's, is a jets study, and the sweeper opens its
# study with load_if_exists, so without the prefix these trials would join the jets studies.
#   1-5    lorentznet_m2  canonical
#   6-10   lorentznet_m2  transformed_lorentz_iso
# A kd_hint search warm-starts every trial from the main-run hint stage at seed 12345 (W0 of
# runlorentznet_m2_train.sh), which must exist first.
#
# Hand-maintained. Every command is commented out; feed the file to
# scripts/cluster/submit_search.sh (olgpu) or submit_search_ngt.sh (a pod), selecting one study
# by its index, since a label (<experiment>_<regime>) is shared by the five studies of a
# regime:
#   bash scripts/cluster/submit_search_ngt.sh --only 1 --workers 4 --trials 20 --seed 1234 \
#       --tag tt_c_kd scripts/top_tagging/rundistil_search.sh
#   bash scripts/cluster/submit_search_ngt.sh --only 4 --workers 4 --trials 60 --seed 1234 \
#       --tag tt_c_rel0 scripts/top_tagging/rundistil_search.sh
# and read each selected point off, within the study's budget, with
#   python3 scripts/search/studies.py best top_tagging_lorentznet_m2_canonical_kd \
#       --db logs/optuna/distillation.db --budget 20

# ======================================================================
# LORENTZNET-M2 CELL
# ======================================================================

# 1. lorentznet_m2  canonical  kd  (top_tagging_lorentznet_m2_canonical_kd)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd algorithm.optimizer.weight_decay=1e-07 data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 2. lorentznet_m2  canonical  kd_hint  (top_tagging_lorentznet_m2_canonical_kd_hint)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd_hint algorithm.optimizer.weight_decay=1e-07 data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 3. lorentznet_m2  canonical  logit_matching  (top_tagging_lorentznet_m2_canonical_logit_matching)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=logit_matching algorithm.optimizer.weight_decay=1e-07 data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 4. lorentznet_m2  canonical  relational alpha 0  (top_tagging_lorentznet_m2_canonical_relational_a0)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational algorithm.optimizer.weight_decay=1e-07 data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_relational_a${algorithm.alpha}' \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 5. lorentznet_m2  canonical  relational alpha 1  (top_tagging_lorentznet_m2_canonical_relational_a1)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational algorithm.optimizer.weight_decay=1e-07 data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=1 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_relational_a${algorithm.alpha}' \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 6. lorentznet_m2  transformed  kd  (top_tagging_lorentznet_m2_transformed_lorentz_iso_kd)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd algorithm.optimizer.weight_decay=1e-07 data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 7. lorentznet_m2  transformed  kd_hint  (top_tagging_lorentznet_m2_transformed_lorentz_iso_kd_hint)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd_hint algorithm.optimizer.weight_decay=1e-07 data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 8. lorentznet_m2  transformed  logit_matching  (top_tagging_lorentznet_m2_transformed_lorentz_iso_logit_matching)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=logit_matching algorithm.optimizer.weight_decay=1e-07 data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_${stage}' \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 9. lorentznet_m2  transformed  relational alpha 0  (top_tagging_lorentznet_m2_transformed_lorentz_iso_relational_a0)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational algorithm.optimizer.weight_decay=1e-07 data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_relational_a${algorithm.alpha}' \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 10. lorentznet_m2  transformed  relational alpha 1  (top_tagging_lorentznet_m2_transformed_lorentz_iso_relational_a1)
# python3 src/train.py -m experiment=top_tagging_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational algorithm.optimizer.weight_decay=1e-07 data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=1 \
#     'hydra.sweeper.study_name=${data.dataset}_${teacher}_${regime_tag}_relational_a${algorithm.alpha}' \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]
