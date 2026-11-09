#!/usr/bin/env bash
# ========================================================================
# MNIST, CIRCULAR CNN CELL: TRAINING COMMANDS
# ========================================================================
# Teacher cnn_circular (mnist_cnn, trained here at the recipe runcnn_search.sh selected and
# net/cnn.yaml and mnist_cnn.yaml record), group translation, student mlp_mnist, alpha_CE 2,
# control teacher cnn_zeros (mnist_cnn_zeros, the zero-padded CNN at the same recipe). The
# teacher-free rows (baseline, label smoothing, augmented) are in runstudent_rows_train.sh.
#
# Waves, in order (04_distillation.md):
#   T   1-6    the two teachers (main run, then folds) and their caches, canonical and
#              transformed; a transformed cache is cut from the canonical teacher run, which
#              teacher_run_dir names in every regime
#   W0  7-8    the hint stage at the fixed hint_lr, warm start of kd_hint and hint_beta
#   S          the searches of rundistil_search.sh; kd_hint's read W0's main run
#   W1  9-18   KD and Hint over T x alpha, logit matching, relational per alpha
#   W2  19-38  after select_hparams.py on W1's main-run rows: headline KD and Hint at the
#              selected T, hint_beta (transformed), control rows
#
# Every training block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition). Folds and extra seeds are submission parameters, and --only takes
# block indices, since a label (<experiment>_<regime>_<method>_main) does not name alpha:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/mnist/runcnn_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/mnist/runcnn_train.sh \
#       callbacks=default
#   bash scripts/cluster/submit_ngt.sh --only 3,4,5,6 --folds 1-5 scripts/mnist/runcnn_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 7,8 scripts/mnist/runcnn_train.sh seed=12346
# [seeds]: also submitted with a trailing seed=12346 and seed=12347, on the main run only. The
# hint blocks take them because a seeded kd_hint warm-starts from the hint run of its seed;
# the teachers never do, since their hp_tag names no seed.
# [folds]: submitted with --folds 1-5 and a trailing callbacks=default, except the cache blocks
# 3-6, which take no trailing override (cache.py composes no callbacks group) and so go in a
# submission of their own. Every grid point goes to the folds, the W1 grids 9, 10, 14, 15
# and the beta grids 27, 28 included, since the T and beta curves carry fold bands (user,
# 9 Nov 2026); the headline blocks 19-26 take the seeds alone. Fold k of a cache
# reads fold k of its teacher, and fold k of kd_hint and hint_beta fold k of the hint stage.
#
# A headline block at seed 12345 is a W1 grid point: train.py finds its manifest and returns.
# Write each T as hyperparameters.csv spells it (4, not 4.0), or the hp_tag names a new
# directory. Edge rule (Block 3): where select_hparams.py flags extend_with, add by hand a
# single-T block at 0.5 or 32 beside the headline block of that channel and alpha, select
# once more, and move the headline to the winner.
#
# Placeholders. A TODO_* token is a string Hydra accepts and the run refuses (an optimiser
# or a loss handed a string raises), so no block trains at a wrong value; replace every
# occurrence of a token, in this file and in runcnn_eval.sh, by the same string.
#   token                        source
#   TODO_LR_KD_{C,X}             studies.py best cnn_circular_<rt>_kd
#   TODO_LR_HINT_{C,X}           studies.py best cnn_circular_<rt>_kd_hint
#   TODO_LR_LOGIT_{C,X}          studies.py best cnn_circular_<rt>_logit_matching
#   TODO_LR_REL_A{0,2}_{C,X}     studies.py best cnn_circular_<rt>_relational_a{0,2}: lr
#   TODO_LIS_A{0,2}_{C,X}        the same trial: lambda_is
#   TODO_LIC_A{0,2}_{C,X}        the same trial: lambda_ic
#   TODO_T_KD_A{0,2}_{C,X}       results/hyperparameters.csv, method kd, alpha 0 or 2
#   TODO_T_HINT_A{0,2}_{C,X}     results/hyperparameters.csv, method hint, alpha 0 or 2
# with <rt> canonical for _C and transformed_translation for _X; studies.py is
# 'python3 scripts/search/studies.py best <study> --db logs/optuna/distillation.db --budget 20'
# (--budget 60 for relational: the workers sharing a study overshoot its trial count), and
# hyperparameters.csv is written by scripts/analysis/select_hparams.py (row: mnist,
# cnn_circular, the regime, the method, the alpha). The control rows take the cell's own
# selected T and lr. The student's weight decay is
# TODO_WD_MNIST in every block, that of Student I's selected student.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# T  TEACHERS AND CACHES
# ======================================================================

# 1. teacher cnn_circular  [folds]
# python3 src/train.py experiment=mnist_cnn method=teacher data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. control teacher cnn_zeros  [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=teacher data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 3. cache cnn_circular  canonical  [folds, no trailing override]
# python3 src/cache.py experiment=mnist_cnn method=cache data.regime=canonical \
#     trainer=gpu trainer.devices=[0]

# 4. cache cnn_circular  transformed  [folds, no trailing override]
# python3 src/cache.py experiment=mnist_cnn method=cache data.regime=transformed \
#     trainer=gpu trainer.devices=[0]

# 5. cache cnn_zeros  canonical  [folds, no trailing override]
# python3 src/cache.py experiment=mnist_cnn_zeros method=cache data.regime=canonical \
#     trainer=gpu trainer.devices=[0]

# 6. cache cnn_zeros  transformed  [folds, no trailing override]
# python3 src/cache.py experiment=mnist_cnn_zeros method=cache data.regime=transformed \
#     trainer=gpu trainer.devices=[0]

# ======================================================================
# W0  HINT STAGE
# ======================================================================

# 7. hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 8. hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W1  GRIDS AND SEARCHED POINTS, after rundistil_search.sh
# ======================================================================

# 9. kd  canonical  T x alpha grid  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 10. kd_hint  canonical  T x alpha grid (exported as method=hint)  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 11. logit_matching  canonical  alpha 0  [folds]
# python3 src/train.py experiment=mnist_cnn method=logit_matching algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 12. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=relational algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_C \
#     algorithm.lambda_ic=TODO_LIC_A0_C algorithm.optimizer.lr=TODO_LR_REL_A0_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 13. relational  canonical  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=relational algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=2 algorithm.lambda_is=TODO_LIS_A2_C \
#     algorithm.lambda_ic=TODO_LIC_A2_C algorithm.optimizer.lr=TODO_LR_REL_A2_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 14. kd  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 15. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 16. logit_matching  transformed  alpha 0  [folds]
# python3 src/train.py experiment=mnist_cnn method=logit_matching algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 17. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=relational algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_X \
#     algorithm.lambda_ic=TODO_LIC_A0_X algorithm.optimizer.lr=TODO_LR_REL_A0_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 18. relational  transformed  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn method=relational algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=2 algorithm.lambda_is=TODO_LIS_A2_X \
#     algorithm.lambda_ic=TODO_LIC_A2_X algorithm.optimizer.lr=TODO_LR_REL_A2_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W2  HEADLINE POINTS, HINT-BETA, CONTROL, after select_hparams.py
# ======================================================================

# 19. kd  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 20. kd  canonical  alpha 2  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A2_C algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 21. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 22. kd_hint  canonical  alpha 2  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A2_C algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 23. kd  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 24. kd  transformed  alpha 2  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A2_X algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 25. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 26. kd_hint  transformed  alpha 2  headline  [seeds]
# python3 src/train.py experiment=mnist_cnn method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 27. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=hint_beta algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 28. hint_beta  transformed  alpha 2  beta grid  [folds]
# python3 src/train.py -m experiment=mnist_cnn method=hint_beta algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 29. control cnn_zeros: hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 30. control cnn_zeros: hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 31. control cnn_zeros: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 32. control cnn_zeros: kd  canonical  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_KD_A2_C algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_KD_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 33. control cnn_zeros: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 34. control cnn_zeros: kd_hint  canonical  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=canonical \
#     algorithm.temperature=TODO_T_HINT_A2_C algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 35. control cnn_zeros: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 36. control cnn_zeros: kd  transformed  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_KD_A2_X algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_KD_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 37. control cnn_zeros: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 38. control cnn_zeros: kd_hint  transformed  alpha 2  [seeds] [folds]
# python3 src/train.py experiment=mnist_cnn_zeros method=kd_hint algorithm.optimizer.weight_decay=TODO_WD_MNIST data.regime=transformed \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
