#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, LLOCA-Z CELL: EVALUATION COMMANDS
# ========================================================================
# One block per trained block of runlloca_z_train.sh and of the [pt, eta, phi] teacher-free
# rows of runstudent_rows_train.sh, with the overrides that name its run directory, so that
# checkpoint_manifest_path resolves by convention. The hint stages are not evaluated: their
# layers above the guided one are still at initialisation. Every block sets data.group=z,
# since the cell's group is the first probe, and the fidelity columns are measured against
# the block's own teacher (lloca_z, jetformer or jetformer_symm_so2).
#
#   1-6    the teachers: lloca_z, the control jetformer and the learned-invariance
#          jetformer_symm_so2, canonical and transformed; the transformed rows read the
#          canonical teacher run (one teacher serves every regime), and the teacher blocks
#          also write the constant-predictor row
#   7-15   the teacher-free rows, evaluated here against lloca_z: baseline, label smoothing,
#          augmented baseline, double-width and four-momentum baselines (trained in
#          jets/none/... with the abs_, abs_wide_ and p4_ prefixes of their hp_tag)
#   16-25  W1, after the W1 training
#   26-63  W2, after the W2 training
#
# Every block evaluates the main run on its validation split alone (eval_splits=[fold]); no
# test row is read before its block of the registration. Folds and seeds follow the training
# catalogue, with eval_splits=[fold,test] appended on fold waves:
#   bash scripts/cluster/submit_ngt.sh --only 1,2 scripts/jets/runlloca_z_eval.sh
#   bash scripts/cluster/submit_ngt.sh --only 7 scripts/jets/runlloca_z_eval.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2 --folds 1-5 scripts/jets/runlloca_z_eval.sh \
#       'eval_splits=[fold,test]'
# [seeds]: main run only, seed=12346 and seed=12347. A headline block (26-33) at seed 12345 on
# the main run is a W1 grid point already evaluated by 16-17 or 21-22, and eval.py appends
# rows unconditionally, so on the main run it is submitted with the extra seeds alone.
# [folds]: --folds 1-5, every point of the T and beta grids included (user, 9 Nov 2026).
#
# Placeholders: the tokens of runlloca_z_train.sh, replaced by the same strings. The
# teacher-free rows are evaluated at the learning rate they trained at, 0.0016805532396569512,
# that of student/jets_pt_eta_phi as its hp_tag spells it: the blocks 7-15 run under
# jets_lloca_z, whose own algorithm.optimizer.lr need not equal it, and the manifest path
# interpolates it.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# TEACHERS
# ======================================================================

# 1. teacher lloca_z  canonical  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=teacher data.regime=canonical \
#     data.group=z eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 2. teacher lloca_z  transformed  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=teacher data.regime=transformed \
#     data.group=z 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 3. control teacher jetformer  canonical  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=canonical \
#     data.group=z eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 4. control teacher jetformer  transformed  [folds]
# python3 src/eval.py experiment=jets_jetformer method=teacher data.regime=transformed \
#     data.group=z 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 5. learned-invariance teacher jetformer_symm_so2  canonical  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=teacher data.regime=canonical \
#     data.group=z eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 6. learned-invariance teacher jetformer_symm_so2  transformed  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=teacher data.regime=transformed \
#     data.group=z 'checkpoint_manifest_path=${teacher_run_dir}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# TEACHER-FREE ROWS (runstudent_rows_train.sh)
# ======================================================================

# 7. baseline  canonical  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=canonical \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 8. baseline  transformed  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=transformed \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 9. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=label_smoothing data.regime=canonical \
#     data.group=z algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 10. label_smoothing  transformed  epsilon curve  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=label_smoothing data.regime=transformed \
#     data.group=z algorithm.label_smoothing=0.05,0.1,0.2 algorithm.optimizer.lr=0.0016805532396569512 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 11. baseline  augmented, on the transformed rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=augmented \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 12. double width: baseline  canonical  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=canonical \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=abs_wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 13. double width: baseline  transformed  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=transformed \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=abs_wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 14. four-momentum: baseline  canonical  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=canonical \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 15. four-momentum: baseline  transformed  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=transformed \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1
# ======================================================================

# 16. kd  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 17. kd_hint  canonical  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 18. logit_matching  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=logit_matching data.regime=canonical data.group=z \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 19. relational  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=relational data.regime=canonical data.group=z \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=26.310383712424244 \
#     algorithm.lambda_ic=9.036384559152717 algorithm.optimizer.lr=0.002174259737446822 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 20. relational  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=relational data.regime=canonical data.group=z \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=15.245652800778585 \
#     algorithm.lambda_ic=4.756815254182737 algorithm.optimizer.lr=0.0032329174033407636 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 21. kd  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 22. kd_hint  transformed  T x alpha grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=1,2,4,8,16 algorithm.alpha=0,1 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 23. logit_matching  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=logit_matching data.regime=transformed data.group=z \
#     algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 24. relational  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=relational data.regime=transformed data.group=z \
#     algorithm.discrepancy=cosine algorithm.alpha=0 algorithm.lambda_is=42.6454004773218 \
#     algorithm.lambda_ic=3.01415850053501 algorithm.optimizer.lr=0.0023276939100527693 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 25. relational  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=relational data.regime=transformed data.group=z \
#     algorithm.discrepancy=cosine algorithm.alpha=1 algorithm.lambda_is=25.092796973147227 \
#     algorithm.lambda_ic=1.538263864840381 algorithm.optimizer.lr=0.0024769868281996885 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W2
# ======================================================================

# 26. kd  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 27. kd  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 28. kd_hint  canonical  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 29. kd_hint  canonical  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 30. kd  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 31. kd  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 32. kd_hint  transformed  alpha 0  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 33. kd_hint  transformed  alpha 1  headline  [seeds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 34. hint_beta  transformed  alpha 0  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=hint_beta data.regime=transformed data.group=z \
#     algorithm.temperature=1 algorithm.alpha=0 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 35. hint_beta  transformed  alpha 1  beta grid  [folds]
# python3 src/eval.py -m experiment=jets_lloca_z method=hint_beta data.regime=transformed data.group=z \
#     algorithm.temperature=2 algorithm.alpha=1 \
#     algorithm.beta=0.05,0.1,0.25,0.5,0.75 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 36. control jetformer: kd  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 37. control jetformer: kd  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 38. control jetformer: kd_hint  canonical  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 39. control jetformer: kd_hint  canonical  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 40. control jetformer: kd  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 41. control jetformer: kd  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 42. control jetformer: kd_hint  transformed  alpha 0  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 43. control jetformer: kd_hint  transformed  alpha 1  [seeds] [folds]
# python3 src/eval.py experiment=jets_jetformer method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 44. learned invariance jetformer_symm_so2: kd  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 45. learned invariance jetformer_symm_so2: kd  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 46. learned invariance jetformer_symm_so2: kd_hint  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 47. learned invariance jetformer_symm_so2: kd_hint  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 48. learned invariance jetformer_symm_so2: kd  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 49. learned invariance jetformer_symm_so2: kd  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 50. learned invariance jetformer_symm_so2: kd_hint  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 51. learned invariance jetformer_symm_so2: kd_hint  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_jetformer_symm_so2 method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.0033465180076422503 \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 52. double width: kd  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 53. double width: kd  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 54. double width: kd_hint  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 55. double width: kd_hint  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=1 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 56. double width: kd  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 57. double width: kd  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 58. double width: kd_hint  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=1 algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 59. double width: kd_hint  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=2 algorithm.alpha=1 algorithm.optimizer.lr=0.0033465180076422503 \
#     variant=wide net@student_net=mlp_jets_abs_wide \
#     'hp_tag=wide_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 60. four-momentum: kd  canonical  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=4 algorithm.alpha=0 algorithm.optimizer.lr=0.0036303760732131728 \
#     variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 61. four-momentum: kd  canonical  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=canonical data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.0036303760732131728 \
#     variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 62. four-momentum: kd  transformed  alpha 0  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=16 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 63. four-momentum: kd  transformed  alpha 1  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=8 algorithm.alpha=1 algorithm.optimizer.lr=0.003144488838025729 \
#     variant=p4 net@student_net=mlp_jets_p4 \
#     'hp_tag=p4_T${algorithm.temperature}_a${algorithm.alpha}_lr${algorithm.optimizer.lr}_s${seed}' \
#     eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# W1 EDGE POINTS, the extra temperature of each arm whose first selection sat on a
# grid end (Block 3; select_hparams.py, 10 Nov 2026), selected once more with it
# ======================================================================

# 64. kd_hint  canonical  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.00032809353377207364 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 65. kd_hint  canonical  T 0.5  alpha 1  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=canonical data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=1 algorithm.optimizer.lr=0.00032809353377207364 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 66. kd_hint  transformed  T 0.5  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd_hint data.regime=transformed data.group=z \
#     algorithm.temperature=0.5 algorithm.alpha=0 algorithm.optimizer.lr=0.0033465180076422503 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# 67. kd  transformed  T 32  alpha 0  edge point  [folds]
# python3 src/eval.py experiment=jets_lloca_z method=kd data.regime=transformed data.group=z \
#     algorithm.temperature=32 algorithm.alpha=0 algorithm.optimizer.lr=0.003144488838025729 eval_splits=[fold] trainer=gpu trainer.devices=[0]

# ======================================================================
# AUGMENTED BASELINE ON THE CANONICAL ROWS (Block 5 of 08_tests.md, 10 Nov 2026)
# ======================================================================
# B and C are read per regime on that regime's rows, so the augmented checkpoints are also
# evaluated on the canonical rows. The run directory takes its own root, since eval_data does
# not enter it and the evaluation on the transformed rows would otherwise be overwritten.

# 68. baseline  augmented, on the canonical rows  [seeds] [folds]
# python3 src/eval.py experiment=jets_lloca_z method=baseline data.regime=augmented \
#     data.group=z algorithm.optimizer.lr=0.0016805532396569512 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     'checkpoint_manifest_path=${paths.log_dir}/${train_task}/${data.dataset}/none/${regime_tag}/${stage}/${run_label}/${hp_tag}/checkpoint_manifest.json' \
#     eval_splits=[fold] eval_data=canonical trainer=gpu trainer.devices=[0] \
#     'hydra.run.dir=${paths.log_dir}/eval_canonical_rows/${run_dir}' \
#     'hydra.sweep.dir=${paths.log_dir}/eval_canonical_rows/${run_dir_parent}'
