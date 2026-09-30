#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS DEEP SETS TRAINING COMMANDS
# ========================================================================
# The stage DAG of one (dataset, teacher) cell, in order, for both regimes:
#   teacher -> cache -> {hint -> {kd_hint, hint_beta, relational(warm)}, kd, relational}
# with baseline independent of all of them. Every block is fold 1; once the
# hyperparameters are selected, re-run the submitter with --folds 2-5 to fill
# the other four folds of the selected points.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit.sh, which queues the blocks on one GPU.

# ======================================================================
# CANONICAL
# ======================================================================

# ----------------------------------------------------------------------
# 1. teacher  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=teacher \
#     data.regime=canonical \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 2. teacher cache  canonical
# logits and hints of the pool and the test split, in pool row order
# ----------------------------------------------------------------------
# python3 src/cache.py \
#     experiment=jets_deepsets \
#     method=cache \
#     data.regime=canonical \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 3. baseline learning-rate scan  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=baseline \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.optimizer.lr=0.001,0.003,0.008,0.01,0.05 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 4. kd temperature x alpha scan  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=kd \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.temperature=1,2,4,8,16 \
#     algorithm.alpha=0,0.1,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 5. hint  canonical
# the warm start of 7 and 11; its own layers above the guided one stay frozen
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=hint \
#     data.regime=canonical \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 6. select the kd point on fold 1, once the evaluations of 3 and 4 are in:
#      python3 scripts/analysis/select_hparams.py results/results.csv
#    then edit the (T, alpha) of the blocks below to the row it prints.

# ----------------------------------------------------------------------
# 7a. kd from the hint checkpoint  canonical
# exported as method=hint (vocabulary.METHOD_OF_STAGE)
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=kd_hint \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 7b. continued hint, beta scan  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=hint_beta \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8a. relational learning-rate scan  canonical
# at lambda_IS 1, lambda_IC 0 and no CE, the cheapest corner of the grid
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=0 \
#     algorithm.optimizer.lr=0.0003,0.001,0.003,0.008,0.01 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8b. relational without CE  canonical
# lambda_IS fixed at 1, so lambda_IC is the ratio of the two terms
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=0 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0,0.125,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8c. relational with CE  canonical
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1,8,32,128 \
#     algorithm.lambda_ic=1,8,32 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 11. relational warm-started from the hint checkpoint  canonical
# hp_tag carries the warm_ prefix, so it lands beside the cold run of the same point
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     'algorithm.warm_start_ckpt=${hint_ckpt}' \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=cosine \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0.25 \
#     hp_tag=warm_${algorithm.discrepancy}_lis${algorithm.lambda_is}_lic${algorithm.lambda_ic}_a${algorithm.alpha}_lr${algorithm.optimizer.lr} \
#     trainer=gpu \
#     trainer.devices=[0]


# ======================================================================
# TRANSFORMED
# ======================================================================

# ----------------------------------------------------------------------
# 1. teacher  transformed
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=teacher \
#     data.regime=transformed \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 2. teacher cache  transformed
# logits and hints of the pool and the test split, in pool row order
# ----------------------------------------------------------------------
# python3 src/cache.py \
#     experiment=jets_deepsets \
#     method=cache \
#     data.regime=transformed \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 3. baseline learning-rate scan  transformed
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=baseline \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.optimizer.lr=0.001,0.003,0.008,0.01,0.05 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 4. kd temperature x alpha scan  transformed
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=kd \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.temperature=1,2,4,8,16 \
#     algorithm.alpha=0,0.1,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 5. hint  transformed
# the warm start of 7 and 11; its own layers above the guided one stay frozen
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=hint \
#     data.regime=transformed \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 6. select the kd point on fold 1, once the evaluations of 3 and 4 are in:
#      python3 scripts/analysis/select_hparams.py results/results.csv
#    then edit the (T, alpha) of the blocks below to the row it prints.

# ----------------------------------------------------------------------
# 7a. kd from the hint checkpoint  transformed
# exported as method=hint (vocabulary.METHOD_OF_STAGE)
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=kd_hint \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 7b. continued hint, beta scan  transformed
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=hint_beta \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8a. relational learning-rate scan  transformed
# at lambda_IS 1, lambda_IC 0 and no CE, the cheapest corner of the grid
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=0 \
#     algorithm.optimizer.lr=0.0003,0.001,0.003,0.008,0.01 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8b. relational without CE  transformed
# lambda_IS fixed at 1, so lambda_IC is the ratio of the two terms
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=0 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0,0.125,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8c. relational with CE  transformed
# ----------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     logger=none \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1,8,32,128 \
#     algorithm.lambda_ic=1,8,32 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 11. relational warm-started from the hint checkpoint  transformed
# hp_tag carries the warm_ prefix, so it lands beside the cold run of the same point
# ----------------------------------------------------------------------
# python3 src/train.py \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     'algorithm.warm_start_ckpt=${hint_ckpt}' \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=cosine \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0.25 \
#     hp_tag=warm_${algorithm.discrepancy}_lis${algorithm.lambda_is}_lic${algorithm.lambda_ic}_a${algorithm.alpha}_lr${algorithm.optimizer.lr} \
#     trainer=gpu \
#     trainer.devices=[0]


# ----------------------------------------------------------------------
# 9. select again on fold 1 from the evaluations of 7 and 8, then
# 10. fill folds 2 to 5 of the selected points:
#      bash scripts/cluster/submit.sh --folds 2-5 --only <labels> <this file>
