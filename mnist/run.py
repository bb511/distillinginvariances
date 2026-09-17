from . import canonical, kd, relational


def main(data_root='data', output_root='paper_outputs/mnist', workflow='both', regime='both', folds=(0, 1, 2, 3, 4), device=None, stages=None, temperatures=(1, 2, 4, 8, 16), kd_output_root=None):
    workflows = ('kd', 'relational') if workflow == 'both' else (workflow,)
    regimes = ('canonical', 'transformed') if regime == 'both' else (regime,)
    for workflow in workflows:
        for regime in regimes:
            if workflow == 'kd':
                if regime == 'canonical':
                    canonical.configure(data_root, output_root, folds, device)
                    selected_stages = stages or ('teacher', 'baseline', 'hint', 'temperature_scan', 'summary')
                    canonical.run(selected_stages, temperatures)
                    continue
                kd.configure(data_root, output_root, regime, folds, device)
                selected_stages = stages or ('teacher', 'baseline', 'hint', 'temperature_scan', 'alpha_temperature_scan', 'summary')
                kd.run(selected_stages, temperatures)
                kd.ALL_X = kd.ALL_Y = None
            else:
                relational.configure(data_root, output_root, regime, folds, device, kd_output_root)
                selected_stages = stages or ('teacher', 'search', 'baseline', 'hint', 'relational', 'search_ce0', 'relational_ce0', 'summary')
                relational.run(selected_stages)
                relational.DATA_CACHE.clear()
