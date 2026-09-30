#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS DEEP SETS EVALUATION COMMANDS
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
# the teacher is its own fidelity reference, so its fidelity columns are 1
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=jets_deepsets \
#     method=teacher \
#     data.regime=canonical \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 3. baseline learning-rate scan  canonical
# one row per point, which is what select_hparams.py reads
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=baseline \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.optimizer.lr=0.001,0.003,0.008,0.01,0.05 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 4. kd temperature x alpha scan  canonical
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=kd \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.temperature=1,2,4,8,16 \
#     algorithm.alpha=0,0.1,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 7a. kd from the hint checkpoint  canonical
# ----------------------------------------------------------------------
# python3 src/eval.py \
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
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=hint_beta \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8a. relational learning-rate scan  canonical
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.alpha=0 \
#     algorithm.optimizer.lr=0.0003,0.001,0.003,0.008,0.01 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8b. relational without CE  canonical
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.alpha=0 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0,0.125,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8c. relational with CE  canonical
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1,8,32,128 \
#     algorithm.lambda_ic=1,8,32 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 11. relational warm-started from the hint checkpoint  canonical
# variant is the provenance column that tells the two relational rows apart
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=canonical \
#     fold=1 \
#     variant=warm_hint \
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
# the teacher is its own fidelity reference, so its fidelity columns are 1
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=jets_deepsets \
#     method=teacher \
#     data.regime=transformed \
#     fold=1 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 3. baseline learning-rate scan  transformed
# one row per point, which is what select_hparams.py reads
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=baseline \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.optimizer.lr=0.001,0.003,0.008,0.01,0.05 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 4. kd temperature x alpha scan  transformed
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=kd \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.temperature=1,2,4,8,16 \
#     algorithm.alpha=0,0.1,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 7a. kd from the hint checkpoint  transformed
# ----------------------------------------------------------------------
# python3 src/eval.py \
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
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=hint_beta \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.temperature=8 \
#     algorithm.alpha=0.25 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8a. relational learning-rate scan  transformed
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.alpha=0 \
#     algorithm.optimizer.lr=0.0003,0.001,0.003,0.008,0.01 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8b. relational without CE  transformed
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.alpha=0 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0,0.125,0.25,0.5,1,2 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 8c. relational with CE  transformed
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     -m \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=mse,cosine,smooth_l1,distance_mse \
#     algorithm.lambda_is=1,8,32,128 \
#     algorithm.lambda_ic=1,8,32 \
#     trainer=gpu \
#     trainer.devices=[0]

# ----------------------------------------------------------------------
# 11. relational warm-started from the hint checkpoint  transformed
# variant is the provenance column that tells the two relational rows apart
# ----------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=jets_deepsets \
#     method=relational \
#     data.regime=transformed \
#     fold=1 \
#     variant=warm_hint \
#     algorithm.alpha=1 \
#     algorithm.discrepancy=cosine \
#     algorithm.lambda_is=1 \
#     algorithm.lambda_ic=0.25 \
#     hp_tag=warm_${algorithm.discrepancy}_lis${algorithm.lambda_is}_lic${algorithm.lambda_ic}_a${algorithm.alpha}_lr${algorithm.optimizer.lr} \
#     trainer=gpu \
#     trainer.devices=[0]


# ----------------------------------------------------------------------
# The hint stage is not evaluated: its layers above the guided one are still at
# their initialisation, so it is no classifier. Folds 2 to 5 of the selected
# points: bash scripts/cluster/submit.sh --folds 2-5 --only <labels> <this file>
