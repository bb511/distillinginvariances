#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, DISTILLATION HYPERPARAMETER SEARCHES
# ========================================================================
# One Optuna study per block, the sibling's protocol (Block 3 of 08_tests.md): TPE seeded at
# 1234 with 20 start-up trials, 50 epochs on the main run, each trial maximising
# val/acc_best, its study row in logs/optuna/distillation.db and nothing else kept (numbered
# trial directories under logs/searches/multiruns/, no logger, no results row). Per cell and
# regime:
#   kd, kd_hint, logit_matching   the learning rate, 20 trials, at T = 4 and alpha 0
#                                 (configs/hparams_search/distillation/lr_optuna.yaml;
#                                 logit_matching does not read T); with 20 start-up trials
#                                 every trial is a seeded random draw
#   relational, per alpha         learning rate, lambda_IS and lambda_IC, 60 trials, cosine
#                                 (relational_optuna.yaml); the warm-up cosine schedule
#                                 spans the 50 epochs, its length following trainer.max_epochs
# The study is <teacher>_<regime_tag>_<stage>, or <teacher>_<regime_tag>_relational_a<alpha>:
#   1-10   deepsets       canonical, transformed_permutation
#   11-20  lorentznet_m2  canonical, transformed_lorentz_iso (jets_lorentznet_m2 does not
#                         exist yet: these blocks fail composition until Teachers I's
#                         re-import lands, and the study takes that experiment's teacher name)
#   21-30  lloca_z        canonical, transformed_z
#   31-40  lloca_so3      canonical, transformed_so3
# A kd_hint search warm-starts every trial from its cell's main-run hint stage at seed 12345
# (W0 of the cell's training catalogue), which must exist first.
#
# Hand-maintained. Every command is commented out; feed the file to
# scripts/cluster/submit_search.sh (olgpu) or submit_search_ngt.sh (a pod), selecting one study
# by its index, since a label (<experiment>_<regime>) is shared by the five studies of a cell
# and regime:
#   bash scripts/cluster/submit_search_ngt.sh --only 1 --workers 4 --trials 20 --seed 1234 \
#       --tag ds_c_kd scripts/jets/rundistil_search.sh
#   bash scripts/cluster/submit_search_ngt.sh --only 4 --workers 4 --trials 60 --seed 1234 \
#       --tag ds_c_rel0 scripts/jets/rundistil_search.sh
# and read each selected point off, within the study's budget, with
#   python3 scripts/search/studies.py best deepsets_canonical_kd \
#       --db logs/optuna/distillation.db --budget 20

# ======================================================================
# DEEP SETS CELL
# ======================================================================

# 1. deepsets  canonical  kd  (deepsets_canonical_kd)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=kd data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 2. deepsets  canonical  kd_hint  (deepsets_canonical_kd_hint)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 3. deepsets  canonical  logit_matching  (deepsets_canonical_logit_matching)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 4. deepsets  canonical  relational alpha 0  (deepsets_canonical_relational_a0)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 5. deepsets  canonical  relational alpha 1  (deepsets_canonical_relational_a1)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 6. deepsets  transformed  kd  (deepsets_transformed_permutation_kd)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=kd data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 7. deepsets  transformed  kd_hint  (deepsets_transformed_permutation_kd_hint)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 8. deepsets  transformed  logit_matching  (deepsets_transformed_permutation_logit_matching)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 9. deepsets  transformed  relational alpha 0  (deepsets_transformed_permutation_relational_a0)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 10. deepsets  transformed  relational alpha 1  (deepsets_transformed_permutation_relational_a1)
# python3 src/train.py -m experiment=jets_deepsets hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# ======================================================================
# LORENTZNET-M2 CELL
# ======================================================================

# 11. lorentznet_m2  canonical  kd  (lorentznet_m2_canonical_kd)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 12. lorentznet_m2  canonical  kd_hint  (lorentznet_m2_canonical_kd_hint)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 13. lorentznet_m2  canonical  logit_matching  (lorentznet_m2_canonical_logit_matching)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 14. lorentznet_m2  canonical  relational alpha 0  (lorentznet_m2_canonical_relational_a0)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 15. lorentznet_m2  canonical  relational alpha 1  (lorentznet_m2_canonical_relational_a1)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 16. lorentznet_m2  transformed  kd  (lorentznet_m2_transformed_lorentz_iso_kd)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 17. lorentznet_m2  transformed  kd_hint  (lorentznet_m2_transformed_lorentz_iso_kd_hint)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 18. lorentznet_m2  transformed  logit_matching  (lorentznet_m2_transformed_lorentz_iso_logit_matching)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 19. lorentznet_m2  transformed  relational alpha 0  (lorentznet_m2_transformed_lorentz_iso_relational_a0)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 20. lorentznet_m2  transformed  relational alpha 1  (lorentznet_m2_transformed_lorentz_iso_relational_a1)
# python3 src/train.py -m experiment=jets_lorentznet_m2 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# ======================================================================
# LLOCA-Z CELL
# ======================================================================

# 21. lloca_z  canonical  kd  (lloca_z_canonical_kd)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=kd data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 22. lloca_z  canonical  kd_hint  (lloca_z_canonical_kd_hint)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 23. lloca_z  canonical  logit_matching  (lloca_z_canonical_logit_matching)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 24. lloca_z  canonical  relational alpha 0  (lloca_z_canonical_relational_a0)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 25. lloca_z  canonical  relational alpha 1  (lloca_z_canonical_relational_a1)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 26. lloca_z  transformed  kd  (lloca_z_transformed_z_kd)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=kd data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 27. lloca_z  transformed  kd_hint  (lloca_z_transformed_z_kd_hint)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 28. lloca_z  transformed  logit_matching  (lloca_z_transformed_z_logit_matching)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 29. lloca_z  transformed  relational alpha 0  (lloca_z_transformed_z_relational_a0)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 30. lloca_z  transformed  relational alpha 1  (lloca_z_transformed_z_relational_a1)
# python3 src/train.py -m experiment=jets_lloca_z hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# ======================================================================
# LLOCA-SO(3) CELL
# ======================================================================

# 31. lloca_so3  canonical  kd  (lloca_so3_canonical_kd)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=kd data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 32. lloca_so3  canonical  kd_hint  (lloca_so3_canonical_kd_hint)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 33. lloca_so3  canonical  logit_matching  (lloca_so3_canonical_logit_matching)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 34. lloca_so3  canonical  relational alpha 0  (lloca_so3_canonical_relational_a0)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 35. lloca_so3  canonical  relational alpha 1  (lloca_so3_canonical_relational_a1)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 36. lloca_so3  transformed  kd  (lloca_so3_transformed_so3_kd)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=kd data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 37. lloca_so3  transformed  kd_hint  (lloca_so3_transformed_so3_kd_hint)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 38. lloca_so3  transformed  logit_matching  (lloca_so3_transformed_so3_logit_matching)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 39. lloca_so3  transformed  relational alpha 0  (lloca_so3_transformed_so3_relational_a0)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 40. lloca_so3  transformed  relational alpha 1  (lloca_so3_transformed_so3_relational_a1)
# python3 src/train.py -m experiment=jets_lloca_so3 hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=1 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]
