from pathlib import Path

import torch


def main(data_root, output_root, workflow='both', regime='both', folds=(0, 1, 2, 3, 4),
         device=None, stages=('teacher', 'baseline', 'hint', 'search', 'distill', 'evaluate'),
         temperatures=(1, 2, 4, 8, 16), relation_search='run'):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    regimes = ('canonical', 'transformed') if regime == 'both' else (regime,)
    output = Path(output_root)
    if workflow in ('kd', 'both'):
        from . import kd
        kd.run(data_root, output / 'lightning', regimes, folds, device, stages, temperatures)
    if workflow in ('relational', 'both'):
        from . import relational
        relational.run(data_root, output / 'notebook_relational', regimes, folds, device,
                       stages, search=relation_search)
