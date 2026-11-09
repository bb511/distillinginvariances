#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, LLOCA-SO(3) CELL: TRAINING COMMANDS
# ========================================================================
# Teacher lloca_so3 (jets_lloca_so3, imported, canonical run and caches in place), group so3,
# student mlp_jets_abs on [pt, eta, phi], alpha_CE 1, control teacher jetformer
# (jets_jetformer, whose own group is z: its transformed blocks set data.group=so3, its
# canonical ones do not, since canonical runs are shared across the cells). The teacher-free
# rows are in runstudent_rows_train.sh.
#
# Waves, in order (04_distillation.md):
#   W0  1-2    the hint stage at the fixed hint_lr, warm start of kd_hint and hint_beta
#   S          the searches of rundistil_search.sh; kd_hint's read W0's main run
#   W1  3-12   KD and Hint over T x alpha, logit matching, relational per alpha
#   W2  13-32  after select_hparams.py on W1's main-run rows: headline KD and Hint at the
#              selected T, hint_beta (transformed), control rows
#
# Every block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition). Folds and extra seeds are submission parameters, and --only takes
# block indices, since a label (<experiment>_<regime>_<method>_main) does not name alpha:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/runlloca_so3_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/runlloca_so3_train.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/runlloca_so3_train.sh \
#       callbacks=default
# [seeds]: also submitted with a trailing seed=12346 and seed=12347, on the main run only. The
# hint blocks take them because a seeded kd_hint warm-starts from the hint run of its seed.
# [folds]: submitted with --folds 1-5 and a trailing callbacks=default. Every grid point goes
# to the folds, the W1 grids 3, 4, 8, 9 and the beta grids 21, 22 included, since the T and
# beta curves carry fold bands (user, 9 Nov 2026); the headline blocks 13-20, whose fold runs
# are the grids', take the seeds alone. Fold k of kd_hint and hint_beta reads fold k of the
# hint stage.
#
# A headline block at seed 12345 is a W1 grid point: train.py finds its manifest and returns.
# Write each T as hyperparameters.csv spells it (4, not 4.0), or the hp_tag names a new
# directory. Edge rule (Block 3): where select_hparams.py flags extend_with, add by hand a
# single-T block at 0.5 or 32 beside the headline block of that channel and alpha, select
# once more, and move the headline to the winner. The canonical control blocks 23 and 25-28
# are, token for token, those of runlorentznet_m2_train.sh and runlloca_z_train.sh: where the
# cells' selected points agree they are one command, and the second submission finds the
# first's manifest and returns.
#
# Placeholders. A TODO_* token is a string Hydra accepts and the run refuses (an optimiser
# or a loss handed a string raises), so no block trains at a wrong value; replace every
# occurrence of a token, in this file and in runlloca_so3_eval.sh, by the same string.
#   token                        source
#   TODO_LR_KD_{C,X}             studies.py best lloca_so3_<rt>_kd
#   TODO_LR_HINT_{C,X}           studies.py best lloca_so3_<rt>_kd_hint
#   TODO_LR_LOGIT_{C,X}          studies.py best lloca_so3_<rt>_logit_matching
#   TODO_LR_REL_A{0,1}_{C,X}     studies.py best lloca_so3_<rt>_relational_a{0,1}: lr
#   TODO_LIS_A{0,1}_{C,X}        the same trial: lambda_is
#   TODO_LIC_A{0,1}_{C,X}        the same trial: lambda_ic
#   TODO_T_KD_A{0,1}_{C,X}       results/hyperparameters.csv, method kd, alpha 0 or 1
#   TODO_T_HINT_A{0,1}_{C,X}     results/hyperparameters.csv, method hint, alpha 0 or 1
# with <rt> canonical for _C and transformed_so3 for _X; studies.py is
# 'python3 scripts/search/studies.py best <study> --db logs/optuna/distillation.db --budget 20'
# (--budget 60 for relational: the workers sharing a study overshoot its trial count), and
# hyperparameters.csv is written by scripts/analysis/select_hparams.py (row: jets, lloca_so3,
# the regime, the method, the alpha). The control rows take the cell's own selected T and lr.
# The student's weight decay is
# 1e-06 in every block, that of the selected student (its comment in configs/net/).
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# W0  HINT STAGE
# ======================================================================

# 1. hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W1  GRIDS AND SEARCHED POINTS, after rundistil_search.sh
# ======================================================================

# 3. kd  canonical  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 4. kd_hint  canonical  T x alpha grid (exported as method=hint)  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 5. logit_matching  canonical  alpha 0  [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=logit_matching algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 6. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=relational algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_C \
#     algorithm.lambda_ic=TODO_LIC_A0_C algorithm.optimizer.lr=TODO_LR_REL_A0_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 7. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=relational algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=TODO_LIS_A1_C \
#     algorithm.lambda_ic=TODO_LIC_A1_C algorithm.optimizer.lr=TODO_LR_REL_A1_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 8. kd  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 9. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 10. logit_matching  transformed  alpha 0  [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=logit_matching algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 11. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=relational algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_X \
#     algorithm.lambda_ic=TODO_LIC_A0_X algorithm.optimizer.lr=TODO_LR_REL_A0_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 12. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_lloca_so3 method=relational algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=TODO_LIS_A1_X \
#     algorithm.lambda_ic=TODO_LIC_A1_X algorithm.optimizer.lr=TODO_LR_REL_A1_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W2  HEADLINE POINTS, HINT-BETA, CONTROL, after select_hparams.py
# ======================================================================

# 13. kd  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 14. kd  canonical  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 15. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 16. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 17. kd  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 18. kd  transformed  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 19. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 20. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_lloca_so3 method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 21. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=hint_beta algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 22. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/train.py -m experiment=jets_lloca_so3 method=hint_beta algorithm.optimizer.weight_decay=1e-06 data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 23. control jetformer: hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 24. control jetformer: hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed data.group=so3 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 25. control jetformer: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 26. control jetformer: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 27. control jetformer: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 28. control jetformer: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 29. control jetformer: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=transformed data.group=so3 \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 30. control jetformer: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd algorithm.optimizer.weight_decay=1e-06 data.regime=transformed data.group=so3 \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 31. control jetformer: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed data.group=so3 \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 32. control jetformer: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_jetformer method=kd_hint algorithm.optimizer.weight_decay=1e-06 data.regime=transformed data.group=so3 \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
