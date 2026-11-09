#!/usr/bin/env bash
# ========================================================================
# MNIST, TEACHER-FREE STUDENT ROWS: EVALUATION COMMANDS
# ========================================================================
# None here. eval.py measures fidelity against a teacher, and the student experiment's teacher
# is none, so each teacher-free row of runstudent_rows_train.sh is evaluated in the MNIST
# cell, under mnist_cnn and the translation group, from the TEACHER-FREE ROWS section of
# scripts/mnist/runcnn_eval.sh, whose blocks name the mnist/none/... run directory through
# their checkpoint_manifest_path.
