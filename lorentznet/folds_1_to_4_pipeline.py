from types import SimpleNamespace
import csv
import json
import statistics
from pathlib import Path
import torch
from .distill.advanced_training import cache_aligned_teacher, train_distillation, train_hint_pretrain, train_supervised
from .distill.evaluation import evaluate_checkpoint
from .distill.lorentz_shuffle import atomic_json, verify_shuffled_dataset
METRICS = ('accu', 'nlll', 'ecel', 'li_jsd', 'li_agree', 'lorentz_relative_logit_error', 'top1_agreement', 'teach_stu_jsd')

def choose_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(value)

def float_tag(value: float) -> str:
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')

class FollowupTaskFactory:

    def __init__(self, config: dict) -> None:
        self.config = config

    def common(self, stage: str, name: str, seed_offset: int=0) -> dict:
        return {'stage': stage, 'name': name, 'fold': int(self.config['fold']), 'n_folds': int(self.config['n_folds']), 'fold_seed': int(self.config['fold_seed']), 'seed': int(self.config['seed']) + seed_offset, 'compute_precision': 'float32_no_tf32'}

    def teacher_task(self, values: dict, index: int) -> dict:
        teacher = self.config['teacher']
        return {**self.common('teacher_fold_followup', values['name'], index), 'kind': 'teacher', **values, 'max_epochs': int(teacher['final_epochs']), 'patience': int(teacher['final_patience']), 'grad_clip': 1.0, 'bn_calibration_samples': int(teacher['bn_calibration_samples']), 'eval_batch_size': int(values['batch_size'])}

    def baseline_task(self) -> dict:
        values = self.config['student']['baseline']
        return {**self.common('mlp_baseline_fold_followup', 'notebook_mlp'), 'kind': 'baseline', 'student_hidden': self.config['student']['hidden'], 'max_epochs': int(values['max_epochs']), 'lr': float(values['lr']), 'weight_decay': float(values['weight_decay']), 'batch_size': int(values['batch_size']), 'eval_batch_size': int(values['batch_size']), 'lr_factor': float(values['lr_factor']), 'lr_patience': int(values['lr_patience']), 'lr_scheduler_mode': values['lr_scheduler_mode'], 'lr_scheduler_monitor': values['lr_scheduler_monitor'], 'patience': int(values['patience']), 'label_smoothing': 0.0, 'l1': 0.0, 'grad_clip': 1.0, 'notebook_reference': "colab_kd_notebook/build_notebook.py CFG['baseline']"}

    def distill_task(self, method: str, stage: str, name: str, index: int, **values) -> dict:
        section_name = 'vrm' if method == 'vrm' else 'kd'
        section = self.config['student'][section_name]
        max_epochs = int(section['max_epochs'])
        task = {**self.common(stage, name, 1000 + index), 'method': method, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': max_epochs, 'patience': int(section['patience']), 'batch_size': int(section['batch_size']), 'eval_batch_size': int(section['batch_size']), 'weight_decay': float(section['weight_decay']), 'l1': 0.0, 'grad_clip': 1.0, **values}
        if method == 'vrm':
            task['warmup_epochs'] = max(1, int(max_epochs * float(section['warmup_fraction'])))
            task['max_pairs'] = int(section['max_pairs'])
            task['scheduler'] = 'warmup_cosine'
        else:
            task['warmup_epochs'] = 0
            task['scheduler'] = 'plateau'
            task['lr_factor'] = float(section['lr_factor'])
            task['lr_patience'] = int(section['lr_patience'])
            task['lr_scheduler_mode'] = section['lr_scheduler_mode']
            task['lr_scheduler_monitor'] = section['lr_scheduler_monitor']
        return task

    def hint_task(self) -> dict:
        section = self.config['student']['hint']
        return {**self.common('hint_pretrain_fold_followup', 'notebook_hint'), 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': int(section['max_epochs']), 'patience': int(section['patience']), 'batch_size': int(section['batch_size']), 'lr': float(section['lr']), 'weight_decay': float(section['weight_decay']), 'warmup_epochs': 0, 'scheduler': 'plateau', 'lr_factor': float(section['lr_factor']), 'lr_patience': int(section['lr_patience']), 'lr_scheduler_mode': section['lr_scheduler_mode'], 'lr_scheduler_monitor': section['lr_scheduler_monitor'], 'notebook_reference': "colab_kd_notebook/build_notebook.py CFG['hint']"}

def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Required fold-0 selection is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))

def _best(path: Path) -> dict:
    value = _json(path)
    best = value.get('best')
    if not isinstance(best, dict):
        raise ValueError(f'Selection file has no best record: {path}')
    if int(best.get('fold', 0)) != 0:
        raise ValueError(f'Selection was not produced on fold 0: {path}')
    return best

def load_fold0_parameters(fold0_root: Path, config: dict) -> dict:
    temperatures = [float(value) for value in config['distillation']['temperatures']]
    teacher_path = fold0_root / '01_teacher' / 'selection.json'
    teacher_best = _best(teacher_path)
    teacher_keys = tuple(config['teacher']['trials'][0].keys())
    teacher = {key: teacher_best[key] for key in teacher_keys}
    teacher['name'] = 'fold0_selected'
    promoted = []
    for method in ('kd', 'hint'):
        for mode in ('no_ce', 'with_ce'):
            root = fold0_root / '04_distillation' / method / mode
            temperature_path = root / 'best_temperature.json'
            chosen = _best(temperature_path)
            ce_weight = 0.0
            sources = [str(temperature_path)]
            if mode == 'with_ce':
                ce_path = root / 'best_ce_weight.json'
                ce_choice = _best(ce_path)
                ce_weight = float(ce_choice['ce_weight'])
                sources.append(str(ce_path))
            for temperature in temperatures:
                promoted.append({'model_key': f'{method}/{mode}/T_{float_tag(temperature)}', 'method': method, 'ce_mode': mode, 'ce_weight': ce_weight, 'temperature': temperature, 'beta': 0.0, 'lr': float(chosen['lr']), 'sources': sources})
    for mode in ('no_ce', 'with_ce'):
        root = fold0_root / '04_distillation' / 'hint_beta' / mode
        temperature_path = root / 'best_temperature.json'
        beta_path = root / 'best_beta.json'
        chosen = _best(temperature_path)
        beta_choice = _best(beta_path)
        sources = [str(temperature_path), str(beta_path)]
        ce_weight = 0.0
        if mode == 'with_ce':
            ce_path = root / 'best_ce_weight.json'
            ce_choice = _best(ce_path)
            ce_weight = float(ce_choice['ce_weight'])
            sources.append(str(ce_path))
        for temperature in temperatures:
            promoted.append({'model_key': f'hint_beta/{mode}/T_{float_tag(temperature)}', 'method': 'hint_beta', 'ce_mode': mode, 'ce_weight': ce_weight, 'temperature': temperature, 'beta': float(beta_choice['beta']), 'lr': float(chosen['lr']), 'sources': sources})
    for criterion in config['distillation']['relation_criteria']:
        for mode in ('no_ce', 'with_ce'):
            root = fold0_root / '04_distillation' / 'vrm' / criterion / mode
            relation_path = root / 'best_relation_parameters.json'
            lr_path = root / 'best_lr.json'
            relation = _best(relation_path)
            lr_choice = _best(lr_path)
            if str(lr_choice['criterion']) != criterion:
                raise ValueError(f'Inconsistent fold-0 relation criterion in {lr_path}')
            promoted.append({'model_key': f'vrm/{criterion}/{mode}', 'method': 'vrm', 'ce_mode': mode, 'ce_weight': 0.0 if mode == 'no_ce' else 1.0, 'criterion': criterion, 'lambda_is': float(relation['lambda_is']), 'lambda_ic': float(relation['lambda_ic']), 'lr': float(lr_choice['lr']), 'sources': [str(relation_path), str(lr_path)]})
    return {'selection_fold': 0, 'temperature_sweep': temperatures, 'temperature_sweep_retrained_each_fold': True, 'fold0_tuning_repeated': False, 'teacher': teacher, 'teacher_source': str(teacher_path), 'promoted_models': promoted}

def _model_key(model: dict) -> str:
    method = str(model.get('method'))
    if method == 'teacher':
        return 'teacher'
    if method == 'baseline':
        return 'baseline'
    if method == 'vrm':
        return f"vrm/{model['criterion']}/{model['ce_mode']}"
    return f"{method}/{model['ce_mode']}/T_{float_tag(float(model['temperature']))}"

class FollowupPipeline:

    def __init__(self, args: SimpleNamespace, config: dict, promoted: dict) -> None:
        self.args = args
        self.config = config
        self.promoted = promoted
        self.output = args.output_root
        self.device = choose_device(args.device)

    @classmethod
    def wait_json(cls, path: Path, context: str, timeout_seconds: int=15 * 60) -> dict:
        return json.loads(path.read_text(encoding='utf-8'))

    def builder(self, fold: int) -> FollowupTaskFactory:
        fold_config = json.loads(json.dumps(self.config))
        fold_config['fold'] = fold
        return FollowupTaskFactory(fold_config)

    def make_jobs(self) -> dict:
        groups = {'teacher': [], 'baseline': [], 'cache': [], 'hint': [], 'distill': [], 'evaluate': []}
        for fold in self.args.folds:
            if fold == 0:
                continue
            fold_root = self.output / f'fold{fold}'
            builder = self.builder(fold)
            teacher_task = builder.teacher_task(self.promoted['teacher'], fold)
            teacher_task['selection_fold'] = 0
            teacher_output = fold_root / '01_teacher' / 'final'
            teacher_job = {'runner': 'supervised', 'task': teacher_task, 'output': teacher_output}
            groups['teacher'].append(teacher_job)
            baseline_task = builder.baseline_task()
            baseline_task['selection_fold'] = 0
            baseline_output = fold_root / '02_mlp_baseline'
            groups['baseline'].append({'runner': 'supervised', 'task': baseline_task, 'output': baseline_output})
            cache_dir = fold_root / '03_teacher_cache'
            groups['cache'].append({'runner': 'cache', 'checkpoint': teacher_output / 'best.pt', 'cache_dir': cache_dir, 'fold': fold})
            hint_task = builder.hint_task()
            hint_task['selection_fold'] = 0
            hint_output = fold_root / '04_distillation' / 'hint_pretrain'
            groups['hint'].append({'runner': 'hint', 'task': hint_task, 'cache_dir': cache_dir, 'output': hint_output})
            fold_distill = []
            for index, chosen in enumerate(self.promoted['promoted_models']):
                method = chosen['method']
                mode = chosen['ce_mode']
                if method == 'vrm':
                    task = builder.distill_task('vrm', f"vrm_{chosen['criterion']}_{mode}_fold_followup", 'final', fold * 100 + index, ce_mode=mode, ce_weight=float(chosen['ce_weight']), criterion=chosen['criterion'], lambda_is=float(chosen['lambda_is']), lambda_ic=float(chosen['lambda_ic']), lr=float(chosen['lr']))
                    output = fold_root / '04_distillation' / 'vrm' / chosen['criterion'] / mode / 'final'
                else:
                    temperature_name = f"T_{float_tag(chosen['temperature'])}"
                    task = builder.distill_task(method, f'{method}_{mode}_fold_followup', temperature_name, fold * 100 + index, ce_mode=mode, ce_weight=float(chosen['ce_weight']), temperature=float(chosen['temperature']), beta=float(chosen['beta']), lr=float(chosen['lr']))
                    output = fold_root / '04_distillation' / method / mode / 'temperature' / temperature_name
                task['selection_fold'] = 0
                task['model_key'] = chosen['model_key']
                fold_distill.append({'runner': 'distill', 'task': task, 'cache_dir': cache_dir, 'hint_checkpoint': hint_output / 'best.pt' if method in ('hint', 'hint_beta') else None, 'output': output})
            groups['distill'].extend(fold_distill)

            def evaluation_job(training_job: dict, model_kind: str, model_key: str, output: Path) -> dict:
                task = training_job['task']
                metadata = {'fold': fold, 'selection_fold': 0, 'model_key': model_key, 'stage': task['stage'], 'name': task['name'], 'method': task.get('method', task.get('kind')), 'ce_mode': task.get('ce_mode'), 'temperature': task.get('temperature'), 'criterion': task.get('criterion'), 'beta': task.get('beta'), 'ce_weight': task.get('ce_weight'), 'lambda_is': task.get('lambda_is'), 'lambda_ic': task.get('lambda_ic'), 'training_output': str(training_job['output'])}
                return {'runner': 'evaluate', 'task': metadata, 'model_kind': model_kind, 'checkpoint': training_job['output'] / 'best.pt', 'cache_dir': cache_dir, 'output': output}
            groups['evaluate'].append(evaluation_job(teacher_job, 'teacher', 'teacher', fold_root / '05_evaluation' / 'teacher'))
            groups['evaluate'].append(evaluation_job(groups['baseline'][-1], 'student', 'baseline', fold_root / '05_evaluation' / 'baseline'))
            for job in fold_distill:
                key = job['task']['model_key']
                groups['evaluate'].append(evaluation_job(job, 'student', key, fold_root / '05_evaluation' / Path(*key.split('/'))))
        return groups

    def run_phase(self, name: str, jobs: list[dict]) -> None:
        for index, job in enumerate(jobs):
            runner = job['runner']
            if runner == 'supervised':
                train_supervised(job['task'], self.args.shuffled_data_dir, job['output'], self.device)
            elif runner == 'cache':
                cache_aligned_teacher(self.args.shuffled_data_dir, job['checkpoint'], job['cache_dir'], self.device, int(self.config['teacher_cache_batch_size']))
            elif runner == 'hint':
                train_hint_pretrain(job['task'], self.args.shuffled_data_dir, job['cache_dir'], job['output'], self.device)
            elif runner == 'distill':
                train_distillation(job['task'], self.args.shuffled_data_dir, job['cache_dir'], job['output'], self.device, job['hint_checkpoint'])
            elif runner == 'evaluate':
                evaluate_checkpoint(job['checkpoint'], job['model_kind'], self.args.shuffled_data_dir, job['cache_dir'], job['output'] / 'evaluation.json', self.device, self.config['evaluation'], job['task'])
            else:
                raise ValueError(f'Unknown runner: {runner}')

    def teacher_results(self, jobs: list[dict]) -> dict:
        path = self.output / '01_teacher_results.json'
        folds = []
        for job in jobs:
            result = self.wait_json(job['output'] / 'result.json', 'teacher result')
            folds.append({'fold': int(job['task']['fold']), 'observed_best_validation_accuracy': float(result['best_val_accuracy']), 'observed_best_validation_nll': float(result['best_val_nll']), 'result': str(job['output'] / 'result.json')})
        atomic_json(path, {'folds': folds})
        return self.wait_json(path, 'teacher results')

    def fold0_evaluations(self) -> list[dict]:
        all_values = _json(self.args.fold0_output_root / '05_summary' / 'all_evaluations.json')
        candidates = all_values.get('evaluations', [])
        selected = []
        wanted = {value['model_key']: value for value in self.promoted['promoted_models']}
        for evaluation in candidates:
            model = evaluation['model']
            key = _model_key(model)
            if key in ('teacher', 'baseline'):
                copied = json.loads(json.dumps(evaluation))
                copied['model']['fold'] = 0
                copied['model']['model_key'] = key
                selected.append(copied)
                continue
            chosen = wanted.get(key)
            if chosen is None:
                continue
            if chosen['method'] == 'vrm':
                match = model.get('category') == 'vrm_final' and str(model.get('criterion')) == chosen['criterion']
            else:
                match = model.get('category') == 'temperature' and abs(float(model.get('temperature')) - float(chosen['temperature'])) < 1e-12 and (abs(float(model.get('ce_weight', 0.0)) - float(chosen['ce_weight'])) < 1e-12) and (abs(float(model.get('beta', 0.0)) - float(chosen['beta'])) < 1e-12)
            if match:
                copied = json.loads(json.dumps(evaluation))
                copied['model']['fold'] = 0
                copied['model']['model_key'] = key
                selected.append(copied)
        expected = 2 + len(wanted)
        keys = [value['model']['model_key'] for value in selected]
        if len(selected) != expected or len(set(keys)) != expected:
            raise ValueError(f'Could not map fold-0 evaluations to promoted models: expected {expected}, got {keys}')
        return selected

    def summarize(self, evaluation_jobs: list[dict], teacher_results: dict) -> None:
        followup = [self.wait_json(job['output'] / 'evaluation.json', 'fold evaluation') for job in evaluation_jobs]
        evaluations = self.fold0_evaluations() + followup
        grouped: dict[str, list[dict]] = {}
        for value in evaluations:
            grouped.setdefault(value['model']['model_key'], []).append(value)
        expected_models = 2 + len(self.promoted['promoted_models'])
        if len(grouped) != expected_models:
            raise ValueError(f'Expected {expected_models} five-fold model groups, got {sorted(grouped)}')
        for key, values in grouped.items():
            folds = sorted((int(value['model']['fold']) for value in values))
            if folds != sorted(self.args.folds):
                raise ValueError(f'Incomplete folds for {key}: {folds}')
        summary_dir = self.output / '05_cv_summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(summary_dir / 'all_fold_evaluations.json', {'evaluations': evaluations})
        fold_columns = ('model_key', 'fold', *METRICS, 'checkpoint')
        with (summary_dir / 'all_fold_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fold_columns)
            writer.writeheader()
            for value in sorted(evaluations, key=lambda item: (item['model']['model_key'], item['model']['fold'])):
                row = {'model_key': value['model']['model_key'], 'fold': value['model']['fold'], **value['metrics'], 'checkpoint': value.get('checkpoint')}
                writer.writerow({key: row.get(key) for key in fold_columns})
        aggregates = []
        for key, values in sorted(grouped.items()):
            metrics = {}
            for metric in METRICS:
                numbers = [float(value['metrics'][metric]) for value in values if value['metrics'].get(metric) is not None]
                metrics[metric] = {'mean': statistics.fmean(numbers) if numbers else None, 'std': statistics.pstdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None, 'count': len(numbers)}
            aggregates.append({'model_key': key, 'folds': sorted(self.args.folds), 'metrics': metrics})
        atomic_json(summary_dir / 'aggregate_metrics.json', {'models': aggregates})
        aggregate_columns = ['model_key']
        for metric in METRICS:
            aggregate_columns.extend((f'{metric}_mean', f'{metric}_std'))
        with (summary_dir / 'aggregate_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
            writer.writeheader()
            for value in aggregates:
                row = {'model_key': value['model_key']}
                for metric in METRICS:
                    row[f'{metric}_mean'] = value['metrics'][metric]['mean']
                    row[f'{metric}_std'] = value['metrics'][metric]['std']
                writer.writerow(row)
        distilled = [value for value in aggregates if value['model_key'] not in ('teacher', 'baseline')]
        best = sorted(distilled, key=lambda value: (-float(value['metrics']['accu']['mean']), float(value['metrics']['nlll']['mean']), value['model_key']))[0]
        summary = {'status': 'complete', 'selection_fold': 0, 'trained_folds': [fold for fold in self.args.folds if fold != 0], 'aggregated_folds': sorted(self.args.folds), 'temperature_sweep': self.promoted['temperature_sweep'], 'temperature_sweep_trained_each_fold': True, 'fold0_tuning_repeated': False, 'model_groups': len(aggregates), 'teacher_training_results': teacher_results, 'best_distilled_model_by_mean_accuracy': best, 'fold_metrics_csv': str(summary_dir / 'all_fold_metrics.csv'), 'aggregate_metrics_csv': str(summary_dir / 'aggregate_metrics.csv'), 'aggregate_metrics_json': str(summary_dir / 'aggregate_metrics.json')}
        atomic_json(self.output / 'summary.json', summary)

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = verify_shuffled_dataset(self.args.shuffled_data_dir)
        fold0_data = _json(self.args.fold0_output_root / '00_data' / 'dataset_reference.json')
        atomic_json(self.output / '00_parameters' / 'fold0_selected_parameters.json', self.promoted)
        atomic_json(self.output / '00_parameters' / 'data_reference.json', {'shuffled_data_dir': str(self.args.shuffled_data_dir), 'fold0_output_root': str(self.args.fold0_output_root)})
        jobs = self.make_jobs()
        self.run_phase('teacher_folds_1_to_4', jobs['teacher'])
        teacher_results = self.teacher_results(jobs['teacher'])
        self.run_phase('baseline_folds_1_to_4', jobs['baseline'])
        self.run_phase('teacher_cache_folds_1_to_4', jobs['cache'])
        self.run_phase('hint_pretrain_folds_1_to_4', jobs['hint'])
        self.run_phase('selected_distillation_folds_1_to_4', jobs['distill'])
        self.run_phase('full_metrics_folds_1_to_4', jobs['evaluate'])
        self.summarize(jobs['evaluate'], teacher_results)
