#!/usr/bin/env bash
# ========================================================================
# TOP TAGGING, LORENTZNET-M2 CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of runlorentznet_m2_train.sh and of the teacher-free rows of
# runstudent_rows_train.sh, with the overrides that name its run directory, so that
# checkpoint_manifest_path resolves by convention. The hint stages are not evaluated: their
# layers above the guided one are still at initialisation. Every block sets
# data.group=lorentz_iso, since the cell's group is the first probe, and the fidelity columns
# are measured against the block's own teacher (the stored-mass LorentzNet, or jetformer for
# the control rows).
#
#   1-2    the teacher, canonical and transformed; the transformed rows read the canonical
#          teacher run (one teacher serves every regime), and the teacher blocks also write
#          the constant-predictor row
#   3-7    the teacher-free rows, evaluated here against the cell's teacher: baseline, label
#          smoothing, augmented baseline (trained in top_tagging/none/...)
#   8-17   W1, after the W1 training
#   18-27  W2, after the W2 training
#   28-29  the control teacher jetformer, canonical and transformed
#   30-37  the control rows, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] and probe_subgroups=[] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/top_tagging/runlorentznet_m2_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 3 scripts/top_tagging/runlorentznet_m2_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/top_tagging/runlorentznet_m2_eval.sh \
#       'eval_splits=[fold,test]' 'probe_subgroups=[]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (18-25) at seed 12345 on
# the main run is a W1 grid point already evaluated by 8-9 or 13-14, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, on the teachers and the rows of Block 8 (08_tests.md, 10 Nov 2026)
# alone; the T and beta grids and the label-smoothing curves are evaluated on the main run. A headline
# block's fold runs are its own, so on folds it is submitted at seed 12345. A fold evaluation
# of this cell appends probe_subgroups=[], which probes the cell's own group, lorentz_iso,
# alone; the subgroup probes run on main-run checkpoints only (Block 8).
#
# Placeholders: the tokens of runlorentznet_m2_train.sh, replaced by the same strings. The
# teacher-free rows are evaluated at the learning rate they trained at, 0.000707422426877928,
# that of student/top_tagging_pt_eta_phi as its hp_tag spells it: the blocks 3-7 run under
# top_tagging_lorentznet_m2, whose own algorithm.optimizer.lr need not equal it, and the
# manifest path interpolates it. Top tagging has a single student input, so these rows carry
# no hp_tag prefix.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHER
# ======================================================================

# 1. teacher  canonical  [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=teacher data.regime=canonical \
#     data.group=lorentz_iso eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher  transformed  [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=teacher data.regime=transformed \
#     data.group=lorentz_iso 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 3. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=baseline data.regime=canonical \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=baseline data.regime=transformed \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 5. label_smoothing  canonical  epsilon curve
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=label_smoothing data.regime=canonical \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. label_smoothing  transformed  epsilon curve
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=label_smoothing data.regime=transformed \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 7. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=baseline data.regime=augmented \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 8. kd  canonical  T x alpha grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. kd_hint  canonical  T x alpha grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 10. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=logit_matching data.regime=canonical data.group=lorentz_iso \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=relational data.regime=canonical data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_C \
#     algorithm.lambda_ic=TODO_LIC_A0_C algorithm.optimizer.lr=TODO_LR_REL_A0_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 12. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=relational data.regime=canonical data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=TODO_LIS_A1_C \
#     algorithm.lambda_ic=TODO_LIC_A1_C algorithm.optimizer.lr=TODO_LR_REL_A1_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. kd  transformed  T x alpha grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. kd_hint  transformed  T x alpha grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=logit_matching data.regime=transformed data.group=lorentz_iso \
#     algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_LOGIT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 16. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=relational data.regime=transformed data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=TODO_LIS_A0_X \
#     algorithm.lambda_ic=TODO_LIC_A0_X algorithm.optimizer.lr=TODO_LR_REL_A0_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=relational data.regime=transformed data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=TODO_LIS_A1_X \
#     algorithm.lambda_ic=TODO_LIC_A1_X algorithm.optimizer.lr=TODO_LR_REL_A1_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 18. kd  canonical  alpha 0  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. kd  canonical  alpha 1  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 20. kd_hint  canonical  alpha 0  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. kd_hint  canonical  alpha 1  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 22. kd  transformed  alpha 0  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. kd  transformed  alpha 1  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. kd_hint  transformed  alpha 0  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. kd_hint  transformed  alpha 1  headline  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 26. hint_beta  transformed  alpha 0  beta grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=hint_beta data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. hint_beta  transformed  alpha 1  beta grid
# python3 src/eval.py -m experiment=top_tagging_lorentznet_m2 method=hint_beta data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# CONTROL TEACHER AND CONTROL ROWS
# ======================================================================

# 28. control teacher jetformer  canonical  [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=teacher data.regime=canonical \
#     data.group=lorentz_iso eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. control teacher jetformer  transformed  [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=teacher data.regime=transformed \
#     data.group=lorentz_iso 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. control jetformer: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. control jetformer: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. control jetformer: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. control jetformer: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_C \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. control jetformer: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_KD_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. control jetformer: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_KD_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control jetformer: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control jetformer: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_jetformer method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=TODO_LR_HINT_X \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# AUGMENTED BASELINE ON THE CANONICAL ROWS (Block 5 of 08_tests.md, 10 Nov 2026)
# ======================================================================
# B and C are read per regime on that regime's rows, so the augmented checkpoints are also
# evaluated on the canonical rows. The run directory takes its own root, since eval_data does
# not enter it and the evaluation on the transformed rows would otherwise be overwritten.

# 38. baseline  augmented, on the canonical rows  [seeds] [folds]
# python3 src/eval.py experiment=top_tagging_lorentznet_m2 method=baseline data.regime=augmented \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.000707422426877928 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] eval_data=canonical trainer=gpu trainer.devices=[0] \
#     'hydra.run.dir=${paths.log_dir}/eval_canonical_rows/${run_dir}' \
#     'hydra.sweep.dir=${paths.log_dir}/eval_canonical_rows/${run_dir_parent}'
