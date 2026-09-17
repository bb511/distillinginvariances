import statistics

def select_fivefold_results(results: list[dict], expected_folds: list[int], expected_groups: int) -> tuple[dict, list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        if result.get('status') != 'complete':
            continue
        key = (str(result['method']), str(result['ce_mode']))
        groups.setdefault(key, []).append(result)
    models = []
    for (method, ce_mode), fold_results in sorted(groups.items()):
        fold_results.sort(key=lambda value: int(value['fold']))
        folds = [int(value['fold']) for value in fold_results]
        if folds != expected_folds:
            raise ValueError(f'Final model {method}/{ce_mode} has folds {folds}, expected {expected_folds}')
        accuracies = [float(value['best_val_accuracy']) for value in fold_results]
        nlls = [float(value['best_val_nll']) for value in fold_results]
        models.append({'method': method, 'ce_mode': ce_mode, 'folds': folds, 'mean_best_val_accuracy': statistics.fmean(accuracies), 'std_best_val_accuracy': statistics.pstdev(accuracies), 'mean_best_val_nll': statistics.fmean(nlls), 'std_best_val_nll': statistics.pstdev(nlls), 'fold_results': fold_results})
    if len(models) != expected_groups:
        raise ValueError(f'Expected {expected_groups} final method/mode groups, found {len(models)}')
    best = sorted(models, key=lambda value: (-float(value['mean_best_val_accuracy']), float(value['mean_best_val_nll']), str(value['method']), str(value['ce_mode'])))[0]
    return (best, models)

def aggregate_fivefold_metrics(evaluations: list[dict], expected_folds: list[int], metric_names: tuple[str, ...]) -> tuple[list[dict], dict]:
    ordered = sorted(evaluations, key=lambda value: int(value['fold']))
    folds = [int(value['fold']) for value in ordered]
    if folds != expected_folds:
        raise ValueError(f'Evaluation folds are {folds}, expected {expected_folds}')
    metrics = {}
    for name in metric_names:
        values = [float(value['metrics'][name]) for value in ordered]
        metrics[name] = {'mean': statistics.fmean(values), 'population_std': statistics.pstdev(values)}
    return (ordered, metrics)
