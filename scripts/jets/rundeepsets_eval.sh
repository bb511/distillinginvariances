#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, DEEP SETS CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of rundeepsets_train.sh and of the teacher-free rows of
# runstudent_rows_train.sh, with the overrides that name its run directory, so that
# checkpoint_manifest_path resolves by convention. The hint stages are not evaluated: their
# layers above the guided one are still at initialisation. Every block sets
# data.group=permutation, since the cell's group is the first probe, and the fidelity columns
# are measured against the block's own teacher (deepsets, or mlp for the control rows).
#
#   1-4    the teachers: deepsets and the control mlp, canonical and transformed; the
#          transformed rows read the canonical teacher run (one teacher serves every regime),
#          and the teacher blocks also write the constant-predictor row
#   5-11   the teacher-free rows, evaluated here against deepsets: baseline, label smoothing,
#          augmented baseline, double-width baseline (trained in jets/none/...)
#   12-21  W1, after the W1 training
#   22-47  W2, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/rundeepsets_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 5 scripts/jets/rundeepsets_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/rundeepsets_eval.sh \
#       'eval_splits=[fold,test]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (22-29) at seed 12345 on
# the main run is a W1 grid point already evaluated by 12-13 or 17-18, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, every point of the T and beta grids included (user, 9 Nov 2026).
#
# Placeholders: the tokens of rundeepsets_train.sh, replaced by the same strings. The
# teacher-free rows are evaluated at the learning rate they trained at, 0.003829873617004707,
# that of student/jets_pt_etarel_phirel as its hp_tag spells it: the blocks 5-11 run under
# jets_deepsets, whose own algorithm.optimizer.lr need not equal it, and the manifest path
# interpolates it.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHERS
# ======================================================================

# 1. teacher deepsets  canonical  [folds]
# python3 src/eval.py experiment=jets_deepsets method=teacher data.regime=canonical \
#     data.group=permutation eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher deepsets  transformed  [folds]
# python3 src/eval.py experiment=jets_deepsets method=teacher data.regime=transformed \
#     data.group=permutation 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 3. control teacher mlp  canonical  [folds]
# python3 src/eval.py experiment=jets_mlp method=teacher data.regime=canonical \
#     data.group=permutation eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. control teacher mlp  transformed  [folds]
# python3 src/eval.py experiment=jets_mlp method=teacher data.regime=transformed \
#     data.group=permutation 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 5. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=baseline data.regime=canonical \
#     data.group=permutation algorithm.optimizer.lr=0.003829873617004707 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=baseline data.regime=transformed \
#     data.group=permutation algorithm.optimizer.lr=0.003829873617004707 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 7. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=label_smoothing data.regime=canonical \
#     data.group=permutation algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.003829873617004707 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 8. label_smoothing  transformed  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=label_smoothing data.regime=transformed \
#     data.group=permutation algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.003829873617004707 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=baseline data.regime=augmented \
#     data.group=permutation algorithm.optimizer.lr=0.003829873617004707 \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 10. double width: baseline  canonical  [folds]
# python3 src/eval.py experiment=jets_deepsets method=baseline data.regime=canonical \
#     data.group=permutation algorithm.optimizer.lr=0.003829873617004707 variant=wide \
#     net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. double width: baseline  transformed  [folds]
# python3 src/eval.py experiment=jets_deepsets method=baseline data.regime=transformed \
#     data.group=permutation algorithm.optimizer.lr=0.003829873617004707 variant=wide \
#     net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 12. kd  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. kd_hint  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=logit_matching data.regime=canonical data.group=permutation \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=relational data.regime=canonical data.group=permutation \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=18.903982689367723 \
#     algorithm.lambda_ic=31.364770582156343 algorithm.optimizer.lr=0.009205793211039868 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 16. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=relational data.regime=canonical data.group=permutation \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=1.1169067443057457 \
#     algorithm.lambda_ic=4.955352821937155 algorithm.optimizer.lr=0.009197356034284456 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. kd  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 18. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=logit_matching data.regime=transformed data.group=permutation \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 20. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=relational data.regime=transformed data.group=permutation \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=94.26949242649104 \
#     algorithm.lambda_ic=0.6298566289253358 algorithm.optimizer.lr=0.005152753924458504 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_deepsets method=relational data.regime=transformed data.group=permutation \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=13.551562490293994 \
#     algorithm.lambda_ic=1.4347984862189431 algorithm.optimizer.lr=0.00801985479779221 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 22. kd  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. kd  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 26. kd  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. kd  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 28. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=hint_beta data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_deepsets method=hint_beta data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. control mlp: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. control mlp: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. control mlp: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. control mlp: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control mlp: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control mlp: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 38. control mlp: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 39. control mlp: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_mlp method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 40. double width: kd  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 41. double width: kd  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 42. double width: kd_hint  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_C algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 43. double width: kd_hint  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=canonical data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_C algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 44. double width: kd  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 45. double width: kd  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_KD_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 46. double width: kd_hint  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A0_X algorithm.alpha=0 algorithm.optimizer.lr=0.009643905397219323 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 47. double width: kd_hint  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_deepsets method=kd_hint data.regime=transformed data.group=permutation \
#     algorithm.temperature=TODO_T_HINT_A1_X algorithm.alpha=1 algorithm.optimizer.lr=0.009643905397219323 \
#     variant=wide net@student_net=mlp_jets_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]
