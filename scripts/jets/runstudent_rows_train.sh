#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, TEACHER-FREE STUDENT ROWS: TRAINING COMMANDS
# ========================================================================
# The rows no teacher enters: the baseline, the label-smoothed students (epsilon 0.05, 0.1,
# 0.2, a reported curve, never selected) and the augmented baseline (the ceiling of Block 2's
# margin and of the transfer fraction), under the student experiments, whose teacher is none,
# so that they train once in jets/none/... and every cell of the same input evaluates them
# (the TEACHER-FREE ROWS of each cell's eval catalogue). Canonical once per input; transformed
# and augmented once per group of that input:
#   [pt, etarel, phirel]  student/jets_pt_etarel_phirel  permutation (DeepSets)         1-7
#   [pt, eta, phi]        student/jets_pt_eta_phi        lorentz_iso, z, so3            8-22
# with the double-width baselines (mlp_jets_wide, canonical and transformed_permutation;
# mlp_jets_abs_wide, canonical and transformed_z) and the four-momentum baseline
# (mlp_jets_p4, canonical and transformed_z). The student's learning rate and weight decay
# are those of its experiment config, filled in once Student I has selected the student.
#
# The two inputs share every canonical stage directory (jets/none/canonical/<stage>/main), so
# the [pt, eta, phi] rows carry an abs_ prefix in their hp_tag, as the variants carry wide_
# and p4_; without it the second input's canonical baseline would meet the first's manifest
# whenever the two learning rates agree. Every transformed and augmented block names its
# group, which sets the regime_tag of its run directory.
#
# Every block is a main-run command at seed 12345 with a checkpoint every 5 epochs
# (callbacks=acquisition), independent of every other: these rows can run at any time.
# Folds and extra seeds are submission parameters, and --only takes block indices, since a
# label (<experiment>_<regime>_<method>_main) names neither the group nor the variant:
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 scripts/jets/runstudent_rows_train.sh
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 scripts/jets/runstudent_rows_train.sh seed=12346
#   bash scripts/cluster/submit_ngt.sh --only 1,2,5 --folds 1-5 scripts/jets/runstudent_rows_train.sh \
#       callbacks=default
# [seeds]: the headline rows, baseline and augmented, also submitted with a trailing
# seed=12346 and seed=12347, on the main run only. [folds]: every block, with --folds 1-5 and
# a trailing callbacks=default.
#
# Every command is commented out: copy one to run it by hand, or feed the file to
# scripts/cluster/submit.sh (olgpu) or submit_ngt.sh (a pod).

# ======================================================================
# [pt, etarel, phirel]
# ======================================================================

# 1. baseline  canonical  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_etarel_phirel method=baseline data.regime=canonical \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 2. baseline  transformed permutation  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_etarel_phirel method=baseline data.regime=transformed \
#     data.group=permutation logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 3. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_etarel_phirel method=label_smoothing data.regime=canonical \
#     algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 4. label_smoothing  transformed permutation  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_etarel_phirel method=label_smoothing data.regime=transformed \
#     data.group=permutation algorithm.label_smoothing=0.05,0.1,0.2 \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 5. baseline  augmented permutation  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_etarel_phirel method=baseline data.regime=augmented \
#     data.group=permutation logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 6. double width: baseline  canonical  [folds]
# python3 src/train.py experiment=student/jets_pt_etarel_phirel method=baseline data.regime=canonical \
#     net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 7. double width: baseline  transformed permutation  [folds]
# python3 src/train.py experiment=student/jets_pt_etarel_phirel method=baseline data.regime=transformed \
#     data.group=permutation net@student_net=mlp_jets_wide 'hp_tag=wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# ======================================================================
# [pt, eta, phi]
# ======================================================================

# 8. baseline  canonical  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=canonical \
#     'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 9. baseline  transformed lorentz_iso  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=transformed \
#     data.group=lorentz_iso 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 10. baseline  transformed z  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=transformed \
#     data.group=z 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 11. baseline  transformed so3  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=transformed \
#     data.group=so3 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 12. label_smoothing  canonical  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_eta_phi method=label_smoothing data.regime=canonical \
#     algorithm.label_smoothing=0.05,0.1,0.2 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 13. label_smoothing  transformed lorentz_iso  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_eta_phi method=label_smoothing data.regime=transformed \
#     data.group=lorentz_iso algorithm.label_smoothing=0.05,0.1,0.2 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 14. label_smoothing  transformed z  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_eta_phi method=label_smoothing data.regime=transformed \
#     data.group=z algorithm.label_smoothing=0.05,0.1,0.2 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 15. label_smoothing  transformed so3  epsilon curve  [folds]
# python3 src/train.py -m experiment=student/jets_pt_eta_phi method=label_smoothing data.regime=transformed \
#     data.group=so3 algorithm.label_smoothing=0.05,0.1,0.2 \
#     'hp_tag=abs_eps${algorithm.label_smoothing}_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 16. baseline  augmented lorentz_iso  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=augmented \
#     data.group=lorentz_iso 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 17. baseline  augmented z  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=augmented \
#     data.group=z 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 18. baseline  augmented so3  [seeds] [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=augmented \
#     data.group=so3 'hp_tag=abs_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 19. double width: baseline  canonical  [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=canonical \
#     net@student_net=mlp_jets_abs_wide 'hp_tag=abs_wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 20. double width: baseline  transformed z  [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=transformed \
#     data.group=z net@student_net=mlp_jets_abs_wide 'hp_tag=abs_wide_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 21. four-momentum: baseline  canonical  [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=canonical \
#     net@student_net=mlp_jets_p4 'hp_tag=p4_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition

# 22. four-momentum: baseline  transformed z  [folds]
# python3 src/train.py experiment=student/jets_pt_eta_phi method=baseline data.regime=transformed \
#     data.group=z net@student_net=mlp_jets_p4 'hp_tag=p4_lr${algorithm.optimizer.lr}_s${seed}' \
#     logger=none trainer=gpu trainer.devices=[0] callbacks=acquisition
