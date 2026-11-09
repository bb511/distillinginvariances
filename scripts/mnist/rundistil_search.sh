#!/usr/bin/env bash
# ========================================================================
# MNIST, DISTILLATION HYPERPARAMETER SEARCHES
# ========================================================================
# One Optuna study per block, the sibling's protocol (Block 3 of 08_tests.md): TPE seeded at
# 1234 with 20 start-up trials, 50 epochs on the main run, each trial maximising
# val/acc_best, its study row in logs/optuna/distillation.db and nothing else kept (numbered
# trial directories under logs/searches/multiruns/, no logger, no results row). Per regime:
#   kd, kd_hint, logit_matching   the learning rate, 20 trials, at T = 4 and alpha 0
#                                 (configs/hparams_search/distillation/lr_optuna.yaml;
#                                 logit_matching does not read T); with 20 start-up trials
#                                 every trial is a seeded random draw
#   relational, alpha 0 and 2     learning rate, lambda_IS and lambda_IC, 60 trials, cosine
#                                 (relational_optuna.yaml); the warm-up cosine schedule
#                                 spans the 50 epochs, its length following trainer.max_epochs
# The study is cnn_circular_<regime_tag>_<stage>, or cnn_circular_<regime_tag>_relational_a<alpha>,
# with <regime_tag> canonical (1-5) or transformed_translation (6-10). A kd_hint search
# warm-starts every trial from the main-run hint stage at seed 12345 (W0 of runcnn_train.sh),
# which must exist first, as must the teacher and its caches.
#
# Hand-maintained. Every command is commented out; feed the file to
# scripts/cluster/submit_search.sh (olgpu) or submit_search_ngt.sh (a pod), selecting one study
# by its index, since a label (<experiment>_<regime>) is shared by the five studies of a
# regime:
#   bash scripts/cluster/submit_search_ngt.sh --only 1 --workers 4 --trials 20 --seed 1234 \
#       --tag mn_c_kd scripts/mnist/rundistil_search.sh
# and read each selected point off, within the study's budget, with
#   python3 scripts/search/studies.py best cnn_circular_canonical_kd \
#       --db logs/optuna/distillation.db --budget 20

# 1. canonical  kd  (cnn_circular_canonical_kd)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=kd data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 2. canonical  kd_hint  (cnn_circular_canonical_kd_hint)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 3. canonical  logit_matching  (cnn_circular_canonical_logit_matching)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=canonical algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 4. canonical  relational alpha 0  (cnn_circular_canonical_relational_a0)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 5. canonical  relational alpha 2  (cnn_circular_canonical_relational_a2)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/relational_optuna \
#     method=relational data.regime=canonical algorithm.discrepancy=cosine algorithm.alpha=2 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 6. transformed  kd  (cnn_circular_transformed_translation_kd)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=kd data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 7. transformed  kd_hint  (cnn_circular_transformed_translation_kd_hint)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=kd_hint data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 8. transformed  logit_matching  (cnn_circular_transformed_translation_logit_matching)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/lr_optuna \
#     method=logit_matching data.regime=transformed algorithm.temperature=4 algorithm.alpha=0 \
#     hydra.sweeper.n_trials=20 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 9. transformed  relational alpha 0  (cnn_circular_transformed_translation_relational_a0)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=0 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]

# 10. transformed  relational alpha 2  (cnn_circular_transformed_translation_relational_a2)
# python3 src/train.py -m experiment=mnist_cnn hparams_search=distillation/relational_optuna \
#     method=relational data.regime=transformed algorithm.discrepancy=cosine algorithm.alpha=2 \
#     hydra.sweeper.n_trials=60 trainer.max_epochs=50 callbacks=search logger=none \
#     trainer=gpu trainer.devices=[0]
