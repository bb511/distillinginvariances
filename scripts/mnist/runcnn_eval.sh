#!/usr/bin/env bash
# ========================================================================
# MNIST, CIRCULAR CNN CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of runcnn_train.sh and of the teacher-free rows of
# runstudent_rows_train.sh, with the overrides that name its run directory, so that
# checkpoint_manifest_path resolves by convention. The hint stages are not evaluated: their
# layers above the guided one are still at initialisation. Every block sets
# data.group=translation, since the cell's group is the first probe, and the fidelity columns
# are measured against the block's own teacher (cnn_circular, or cnn_zeros for the control
# rows).
#
#   1-4    the teachers: cnn_circular and the control cnn_zeros, canonical and transformed;
#          the transformed rows read the canonical teacher run (one teacher serves every
#          regime), and the teacher blocks also write the constant-predictor row
#   5-9    the teacher-free rows, evaluated here against cnn_circular: baseline, label
#          smoothing, augmented baseline (trained in mnist/none/...)
#   10-19  W1, after the W1 training
#   20-37  W2, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/mnist/runcnn_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 5 scripts/mnist/runcnn_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/mnist/runcnn_eval.sh \
#       'eval_splits=[fold,test]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (20-27) at seed 12345 on
# the main run is a W1 grid point already evaluated by 10-11 or 15-16, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, every point of the T and beta grids included (user, 9 Nov 2026).
#
# Placeholders: the tokens of runcnn_train.sh, replaced by the same strings. The teacher-free
# rows are evaluated at the learning rate they trained at, 0.004491388971536795, that of
# student/mnist as its hp_tag spells it: the blocks 5-9 run under mnist_cnn, whose own
# algorithm.optimizer.lr need not equal it, and the manifest path interpolates it.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHERS
# ======================================================================

# 1. teacher cnn_circular  canonical  [folds]
# python3 src/eval.py experiment=mnist_cnn method=teacher data.regime=canonical \
#     data.group=translation eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher cnn_circular  transformed  [folds]
# python3 src/eval.py experiment=mnist_cnn method=teacher data.regime=transformed \
#     data.group=translation 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 3. control teacher cnn_zeros  canonical  [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=teacher data.regime=canonical \
#     data.group=translation eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. control teacher cnn_zeros  transformed  [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=teacher data.regime=transformed \
#     data.group=translation 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 5. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=baseline data.regime=canonical \
#     data.group=translation algorithm.optimizer.lr=0.004491388971536795 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=baseline data.regime=transformed \
#     data.group=translation algorithm.optimizer.lr=0.004491388971536795 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 7. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=label_smoothing data.regime=canonical \
#     data.group=translation algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.004491388971536795 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 8. label_smoothing  transformed  epsilon curve  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=label_smoothing data.regime=transformed \
#     data.group=translation algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.004491388971536795 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=baseline data.regime=augmented \
#     data.group=translation algorithm.optimizer.lr=0.004491388971536795 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 10. kd  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=kd data.regime=canonical data.group=translation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. kd_hint  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=kd_hint data.regime=canonical data.group=translation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 12. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=mnist_cnn method=logit_matching data.regime=canonical data.group=translation \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=relational data.regime=canonical data.group=translation \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=1.8719208042450481 \
#     algorithm.lambda_ic=22.830195410660377 algorithm.optimizer.lr=0.0073020857975766695 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. relational  canonical  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=relational data.regime=canonical data.group=translation \
#     algorithm.discrepancy=cosine algorithm.alpha=2 algorithm.lambda_is=0.5126685560125409 \
#     algorithm.lambda_ic=12.659652847015726 algorithm.optimizer.lr=0.00994036098299559 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. kd  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=kd data.regime=transformed data.group=translation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=0.0037214926657368383 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 16. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=kd_hint data.regime=transformed data.group=translation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,2 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=mnist_cnn method=logit_matching data.regime=transformed data.group=translation \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 18. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=relational data.regime=transformed data.group=translation \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=131.1210640602283 \
#     algorithm.lambda_ic=10.421322042642588 algorithm.optimizer.lr=0.003645917883913076 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. relational  transformed  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn method=relational data.regime=transformed data.group=translation \
#     algorithm.discrepancy=cosine algorithm.alpha=2 algorithm.lambda_is=1.6927389645255555 \
#     algorithm.lambda_ic=6.421207563312898 algorithm.optimizer.lr=0.00934575380356406 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 20. kd  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. kd  canonical  alpha 2  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_KD_A2_C algorithm.alpha=2 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 22. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd_hint data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. kd_hint  canonical  alpha 2  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd_hint data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A2_C algorithm.alpha=2 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. kd  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.0037214926657368383 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. kd  transformed  alpha 2  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_KD_A2_X algorithm.alpha=2 algorithm.optimizer.lr=0.0037214926657368383 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 26. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd_hint data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. kd_hint  transformed  alpha 2  headline  [seeds]
# python3 src/eval.py experiment=mnist_cnn method=kd_hint data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 28. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=hint_beta data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. hint_beta  transformed  alpha 2  beta grid  [folds]
# python3 src/eval.py -m experiment=mnist_cnn method=hint_beta data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. control cnn_zeros: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. control cnn_zeros: kd  canonical  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_KD_A2_C algorithm.alpha=2 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. control cnn_zeros: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd_hint data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. control cnn_zeros: kd_hint  canonical  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd_hint data.regime=canonical data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A2_C algorithm.alpha=2 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. control cnn_zeros: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.0037214926657368383 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. control cnn_zeros: kd  transformed  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_KD_A2_X algorithm.alpha=2 algorithm.optimizer.lr=0.0037214926657368383 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control cnn_zeros: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd_hint data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control cnn_zeros: kd_hint  transformed  alpha 2  [seeds] [folds]
# python3 src/eval.py experiment=mnist_cnn_zeros method=kd_hint data.regime=transformed data.group=translation \
#     algorithm.temperature=TODO_T_HINT_A2_X algorithm.alpha=2 algorithm.optimizer.lr=0.008082117583015577 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]
