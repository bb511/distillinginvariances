#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, LLOCA-SO(3) CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of runlloca_so3_train.sh and of the [pt, eta, phi]
# teacher-free rows of runstudent_rows_train.sh, with the overrides that name its run
# directory, so that checkpoint_manifest_path resolves by convention. The hint stages are not
# evaluated: their layers above the guided one are still at initialisation. Every block sets
# data.group=so3, since the cell's group is the first probe, and the fidelity columns are
# measured against the block's own teacher (lloca_so3, or jetformer for the control rows).
#
#   1-4    the teachers: lloca_so3 and the control jetformer, canonical and transformed; the
#          transformed rows read the canonical teacher run (one teacher serves every regime),
#          and the teacher blocks also write the constant-predictor row
#   5-9    the teacher-free rows, evaluated here against lloca_so3: baseline, label
#          smoothing, augmented baseline (trained in jets/none/... with the abs_ prefix of
#          their hp_tag)
#   10-19  W1, after the W1 training
#   20-37  W2, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/runlloca_so3_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 5 scripts/jets/runlloca_so3_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/runlloca_so3_eval.sh \
#       'eval_splits=[fold,test]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (20-27) at seed 12345 on
# the main run is a W1 grid point already evaluated by 10-11 or 15-16, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, every point of the T and beta grids included (user, 9 Nov 2026).
#
# Placeholders: the tokens of runlloca_so3_train.sh, replaced by the same strings. The
# teacher-free rows are evaluated at the learning rate they trained at, 0.0016805532396569512,
# that of student/jets_pt_eta_phi as its hp_tag spells it: the blocks 5-9 run under
# jets_lloca_so3, whose own algorithm.optimizer.lr need not equal it, and the manifest path
# interpolates it.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHERS
# ======================================================================

# 1. teacher lloca_so3  canonical  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=teacher data.regime=canonical \
#     data.group=so3 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher lloca_so3  transformed  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=teacher data.regime=transformed \
#     data.group=so3 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 3. control teacher jetformer  canonical  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=canonical \
#     data.group=so3 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. control teacher jetformer  transformed  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=transformed \
#     data.group=so3 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 5. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=baseline data.regime=canonical \
#     data.group=so3 algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=baseline data.regime=transformed \
#     data.group=so3 algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 7. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=label_smoothing data.regime=canonical \
#     data.group=so3 algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 8. label_smoothing  transformed  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=label_smoothing data.regime=transformed \
#     data.group=so3 algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=baseline data.regime=augmented \
#     data.group=so3 algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 10. kd  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. kd_hint  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 12. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=logit_matching data.regime=canonical data.group=so3 \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.003682506413773172 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=relational data.regime=canonical data.group=so3 \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=32.18096890995607 \
#     algorithm.lambda_ic=4.28023949608788 algorithm.optimizer.lr=0.0019840017265240492 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=relational data.regime=canonical data.group=so3 \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=9.365475629437169 \
#     algorithm.lambda_ic=1.0397067242195943 algorithm.optimizer.lr=0.0018216142138660274 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. kd  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 16. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=logit_matching data.regime=transformed data.group=so3 \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 18. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=relational data.regime=transformed data.group=so3 \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=3.4790949906977033 \
#     algorithm.lambda_ic=7.966680894352464 algorithm.optimizer.lr=0.00524834009684616 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=relational data.regime=transformed data.group=so3 \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=20.999356435866904 \
#     algorithm.lambda_ic=0.7794960572427161 algorithm.optimizer.lr=0.009425564849420557 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 20. kd  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. kd  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 22. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. kd  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. kd  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 26. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 28. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=hint_beta data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_so3 method=hint_beta data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. control jetformer: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. control jetformer: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. control jetformer: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. control jetformer: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. control jetformer: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. control jetformer: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=16 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control jetformer: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control jetformer: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1 EDGE POINTS, the extra temperature of each arm whose first selection sat on a
# grid end (Block 3; select_hparams.py, 10 Nov 2026), selected once more with it
# ======================================================================

# 38. kd_hint  canonical  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.001754759256702473 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 39. kd_hint  canonical  T 32  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=canonical data.group=so3 \
#     algorithm.temperature=32 algorithm.alpha=1 algorithm.optimizer.lr=0.001754759256702473 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 40. kd  canonical  T 32  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=canonical data.group=so3 \
#     algorithm.temperature=32 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 41. kd_hint  transformed  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 42. kd_hint  transformed  T 0.5  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd_hint data.regime=transformed data.group=so3 \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 43. kd  transformed  T 32  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=kd data.regime=transformed data.group=so3 \
#     algorithm.temperature=32 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# AUGMENTED BASELINE ON THE CANONICAL ROWS (Block 5 of 08_tests.md, 10 Nov 2026)
# ======================================================================
# B and C are read per regime on that regime's rows, so the augmented checkpoints are also
# evaluated on the canonical rows. The run directory takes its own root, since eval_data does
# not enter it and the evaluation on the transformed rows would otherwise be overwritten.

# 44. baseline  augmented, on the canonical rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_so3 method=baseline data.regime=augmented \
#     data.group=so3 algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] eval_data=canonical trainer=gpu trainer.devices=[0] \
#     'hydra.run.dir=${paths.log_dir}/eval_canonical_rows/${run_dir}' \
#     'hydra.sweep.dir=${paths.log_dir}/eval_canonical_rows/${run_dir_parent}'
