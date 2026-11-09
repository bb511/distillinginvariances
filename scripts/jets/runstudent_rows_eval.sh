#!/usr/bin/env bash
# ========================================================================
# HLS4ML JETS, TEACHER-FREE STUDENT ROWS: EVALUATION COMMANDS
# ========================================================================
# None here. eval.py measures fidelity against a teacher, and the student experiments' teacher
# is none, so each teacher-free row of runstudent_rows_train.sh is evaluated in every cell of
# its input, under the cell's experiment and group, from the TEACHER-FREE ROWS section of:
#   [pt, etarel, phirel]  scripts/jets/rundeepsets_eval.sh
#   [pt, eta, phi]        scripts/jets/runlorentznet_m2_eval.sh, runlloca_z_eval.sh,
#                         runlloca_so3_eval.sh
# each of whose blocks names the jets/none/... run directory through its
# checkpoint_manifest_path.
