#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, LORENTZNET-M2 CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of runlorentznet_m2_train.sh and of the [pt, eta, phi]
# teacher-free rows of runstudent_rows_train.sh, with the overrides that name its run
# directory, so that checkpoint_manifest_path resolves by convention. The hint stages are not
# evaluated: their layers above the guided one are still at initialisation. Every block sets
# data.group=lorentz_iso, since the cell's group is the first probe, and the fidelity columns
# are measured against the block's own teacher (the stored-mass LorentzNet, or jetformer for
# the control rows). jets_lorentznet_m2 does not exist yet, so its blocks fail composition
# until Teachers I's re-import lands.
#
#   1-4    the teachers: the stored-mass LorentzNet and the control jetformer, canonical
#          and transformed; the transformed rows read the canonical teacher run (one teacher
#          serves every regime), and the teacher blocks also write the constant-predictor row
#   5-9    the teacher-free rows, evaluated here against the cell's teacher: baseline, label
#          smoothing, augmented baseline (trained in jets/none/... with the abs_ prefix of
#          their hp_tag)
#   10-19  W1, after the W1 training
#   20-37  W2, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/runlorentznet_m2_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 5 scripts/jets/runlorentznet_m2_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/runlorentznet_m2_eval.sh \
#       'eval_splits=[fold,test]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (20-27) at seed 12345 on
# the main run is a W1 grid point already evaluated by 10-11 or 15-16, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, every point of the T and beta grids included (user, 9 Nov 2026).
#
# Placeholders: the tokens of runlorentznet_m2_train.sh, replaced by the same strings. The
# teacher-free rows are evaluated at the learning rate they trained at, 0.0016805532396569512,
# that of student/jets_pt_eta_phi as its hp_tag spells it: the blocks 5-9 run under
# jets_lorentznet_m2, whose own algorithm.optimizer.lr need not equal it, and the manifest path
# interpolates it.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHERS
# ======================================================================

# 1. teacher  canonical  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=teacher data.regime=canonical \
#     data.group=lorentz_iso eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher  transformed  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=teacher data.regime=transformed \
#     data.group=lorentz_iso 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 3. control teacher jetformer  canonical  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=canonical \
#     data.group=lorentz_iso eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. control teacher jetformer  transformed  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=transformed \
#     data.group=lorentz_iso 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 5. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=baseline data.regime=canonical \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=baseline data.regime=transformed \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 7. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=label_smoothing data.regime=canonical \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 8. label_smoothing  transformed  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=label_smoothing data.regime=transformed \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=baseline data.regime=augmented \
#     data.group=lorentz_iso algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 10. kd  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. kd_hint  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 12. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=logit_matching data.regime=canonical data.group=lorentz_iso \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=relational data.regime=canonical data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=45.60011257235895 \
#     algorithm.lambda_ic=9.034612790026392 algorithm.optimizer.lr=0.0020556652997365263 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=relational data.regime=canonical data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=3.437903390231893 \
#     algorithm.lambda_ic=0.8657311805789505 algorithm.optimizer.lr=0.0018121196716329618 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. kd  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 16. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=logit_matching data.regime=transformed data.group=lorentz_iso \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 18. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=relational data.regime=transformed data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=29.96075663081402 \
#     algorithm.lambda_ic=0.9403538994851475 algorithm.optimizer.lr=0.0016746863339222257 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=relational data.regime=transformed data.group=lorentz_iso \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=161.83909664448743 \
#     algorithm.lambda_ic=0.5753119628930268 algorithm.optimizer.lr=0.006026183188595793 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 20. kd  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. kd  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 22. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. kd  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=2 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. kd  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 26. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 28. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=hint_beta data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lorentznet_m2 method=hint_beta data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. control jetformer: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. control jetformer: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. control jetformer: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. control jetformer: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. control jetformer: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=2 algorithm.alpha=0 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. control jetformer: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0027625564527090462 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control jetformer: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control jetformer: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1 EDGE POINTS, the extra temperature of each arm whose first selection sat on a
# grid end (Block 3; select_hparams.py, 10 Nov 2026), selected once more with it
# ======================================================================

# 38. kd_hint  canonical  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.0009403641842918084 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 39. kd_hint  canonical  T 0.5  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.0009403641842918084 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 40. kd  canonical  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 41. kd  canonical  T 0.5  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd data.regime=canonical data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 42. kd_hint  transformed  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 43. kd_hint  transformed  T 0.5  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lorentznet_m2 method=kd_hint data.regime=transformed data.group=lorentz_iso \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 eval_splits=[fold] trainer=gpu trainer.devices=[0]
