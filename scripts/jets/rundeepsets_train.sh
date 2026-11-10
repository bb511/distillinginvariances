#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, DEEP SETS CELL: TRAINING COMMANDS
# ========================================================================
# Teacher deepsets (jets_deepsets, imported, canonical run and caches in place), group
# permutation, student mlp_jets on [pt, etarel, phirel], alpha_CE 1, control teacher mlp
# (jets_mlp), double-width student mlp_jets_wide. The teacher-free rows (baseline, label
# smoothing, augmented, double-width baseline) are in runstudent_rows_train.sh.
#
# Waves, in order (04_distillation.md):
#   W0  1-2    the hint stage at the fixed hint_lr, warm start of kd_hint and hint_beta
#   S          the searches of rundistil_search.sh; kd_hint's read W0's main run
#   W1  3-12   KD and Hint over T x alpha, logit matching, relational per alpha
#   W2  13-42  after select_hparams.py on W1's main-run rows: headline KD and Hint at the
#              selected T, hint_beta (transformed), control rows, double-width rows
#
# Every block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition). Folds and extra seeds are submission parameters, and --only takes
# block indices, since a label (<experiment>_<regime>_<method>_main) names neither alpha nor
# variant:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/rundeepsets_train.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/rundeepsets_train.sh \
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
# once more, and move the headline to the winner.
#
# Placeholders. A TODO_* token is a string Hydra accepts and the run refuses (an optimiser
# or a loss handed a string raises), so no block trains at a wrong value; replace every
# occurrence of a token, in this file and in rundeepsets_eval.sh, by the same string.
#   token                        source
#   TODO_LR_KD_{C,X}             studies.py best deepsets_<rt>_kd
#   TODO_LR_HINT_{C,X}           studies.py best deepsets_<rt>_kd_hint
#   TODO_LR_LOGIT_{C,X}          studies.py best deepsets_<rt>_logit_matching
#   TODO_LR_REL_A{0,1}_{C,X}     studies.py best deepsets_<rt>_relational_a{0,1}: lr
#   TODO_LIS_A{0,1}_{C,X}        the same trial: lambda_is
#   TODO_LIC_A{0,1}_{C,X}        the same trial: lambda_ic
#   TODO_T_KD_A{0,1}_{C,X}       results/hyperparameters.csv, method kd, alpha 0 or 1
#   TODO_T_HINT_A{0,1}_{C,X}     results/hyperparameters.csv, method hint, alpha 0 or 1
# with <rt> canonical for _C and transformed_permutation for _X; studies.py is
# 'python3 scripts/search/studies.py best <study> --db logs/optuna/distillation.db --budget 20'
# (--budget 60 for relational: the workers sharing a study overshoot its trial count), and
# hyperparameters.csv is written by scripts/analysis/select_hparams.py (row: jets, deepsets,
# the regime, the method, the alpha). The control and double-width rows take the cell's own
# selected T and lr. The student's weight decay is
# 0.0 in every block, that of the selected student (its comment in configs/net/).
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# W0  HINT STAGE
# ======================================================================

# 1. hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W1  GRIDS AND SEARCHED POINTS, after rundistil_search.sh
# ======================================================================

# 3. kd  canonical  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0027625564527090462 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 4. kd_hint  canonical  T x alpha grid (exported as method=hint)  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.003144488838025729 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 5. logit_matching  canonical  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=logit_matching algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 6. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=relational algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=18.903982689367723 \
#     algorithm.lambda_ic=31.364770582156343 algorithm.optimizer.lr=0.009205793211039868 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 7. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=relational algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=1.1169067443057457 \
#     algorithm.lambda_ic=4.955352821937155 algorithm.optimizer.lr=0.009197356034284456 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 8. kd  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.001754759256702473 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 9. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 10. logit_matching  transformed  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=logit_matching algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 11. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=relational algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=94.26949242649104 \
#     algorithm.lambda_ic=0.6298566289253358 algorithm.optimizer.lr=0.005152753924458504 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 12. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_deepsets method=relational algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=13.551562490293994 \
#     algorithm.lambda_ic=1.4347984862189431 algorithm.optimizer.lr=0.00801985479779221 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W2  HEADLINE POINTS, HINT-BETA, CONTROL, DOUBLE WIDTH, after select_hparams.py
# ======================================================================

# 13. kd  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=8 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 14. kd  canonical  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=4 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 15. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 16. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 17. kd  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 18. kd  transformed  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 19. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 20. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 21. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=hint_beta algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 22. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/train.py -m experiment=jets_deepsets method=hint_beta algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=16 algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 23. control mlp: hint  canonical  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 24. control mlp: hint  transformed  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 25. control mlp: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=8 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 26. control mlp: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=4 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 27. control mlp: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 28. control mlp: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 29. control mlp: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 30. control mlp: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 31. control mlp: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 32. control mlp: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/train.py experiment=jets_mlp method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 33. double width: hint  canonical  [folds]
# python3 src/train.py experiment=jets_deepsets method=hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 34. double width: hint  transformed  [folds]
# python3 src/train.py experiment=jets_deepsets method=hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 35. double width: kd  canonical  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=8 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 36. double width: kd  canonical  alpha 1  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=4 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 37. double width: kd_hint  canonical  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     net@student_net=mlp_jets_wide 'hint_hp_tag=wide_lr${hint_lr}_s${seed}' \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 38. double width: kd_hint  canonical  alpha 1  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     net@student_net=mlp_jets_wide 'hint_hp_tag=wide_lr${hint_lr}_s${seed}' \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 39. double width: kd  transformed  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 40. double width: kd  transformed  alpha 1  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 41. double width: kd_hint  transformed  alpha 0  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     net@student_net=mlp_jets_wide 'hint_hp_tag=wide_lr${hint_lr}_s${seed}' \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 42. double width: kd_hint  transformed  alpha 1  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     net@student_net=mlp_jets_wide 'hint_hp_tag=wide_lr${hint_lr}_s${seed}' \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# W1 EDGE POINTS, the extra temperature of each arm whose first selection sat on a
# grid end (Block 3; select_hparams.py, 10 Nov 2026), selected once more with it
# ======================================================================

# 43. kd_hint  canonical  T 0.5  alpha 0  edge point  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 44. kd_hint  canonical  T 0.5  alpha 1  edge point  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=canonical \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 45. kd_hint  transformed  T 0.5  alpha 0  edge point  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 46. kd_hint  transformed  T 32  alpha 1  edge point  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd_hint algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=32 algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 47. kd  transformed  T 0.5  alpha 0  edge point  [folds]
# python3 src/train.py experiment=jets_deepsets method=kd algorithm.optimizer.weight_decay=0.0 data.regime=transformed \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
