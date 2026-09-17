from types import SimpleNamespace
import csv
import json
from pathlib import Path
import torch
from .distill.advanced_training import cache_aligned_teacher, select_best, strict_fp32, teacher_checkpoint_config, train_distillation, train_hint_pretrain, train_supervised, write_results_csv
from .distill.aligned_data import verify_aligned_dataset
from .distill.aligned_evaluation import evaluate_best_student, evaluate_teacher
from .distill.fivefold import aggregate_fivefold_metrics, select_fivefold_results
from .distill.lorentz_shuffle import atomic_json
ROOT = Path(__file__).resolve().parent

def choose_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(value)

def float_tag(value: float) -> str:
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')

class Pipeline:

    def __init__(self, args: SimpleNamespace, config: dict) -> None:
        self.args = args
        self.config = config
        self.output = args.output_root
        self.device = choose_device(args.device)
        strict_fp32()
        self.folds = list(args.folds)
        self.teacher_checkpoints = {fold: self.teacher_checkpoint_path(args.teacher_root, fold) for fold in self.folds}
        self.teacher_configs = {fold: teacher_checkpoint_config(checkpoint) for fold, checkpoint in self.teacher_checkpoints.items()}
        self.fold_seed = int(self.teacher_configs[0].get('fold_seed', 0))
        self.n_folds = 5

    @staticmethod
    def teacher_checkpoint_path(root: Path, fold: int) -> Path:
        if fold == 0:
            return root / 'final' / 'best_fp32_full' / 'best.pt'
        return root / 'final_cv5' / f'best_fp32_fold{fold}' / 'best.pt'

    def cache_dir(self, fold: int) -> Path:
        return self.output / '01_teacher_cache' / f'fold{fold}'

    def hint_checkpoint(self, fold: int) -> Path:
        return self.output / '02_hint_pretrain' / f'fold{fold}' / 'best.pt'

    def common(self, stage: str, name: str, fold: int=0) -> dict:
        return {'stage': stage, 'name': name, 'fold': int(fold), 'n_folds': self.n_folds, 'fold_seed': self.fold_seed, 'compute_precision': 'float32_no_tf32'}

    def hint_task(self, fold: int) -> dict:
        section = self.config['student']['hint']
        return {**self.common('hint_pretrain', 'notebook_hint', fold), 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'notebook_guided_module_index': int(self.config['student']['notebook_guided_module_index']), 'max_epochs': int(section['max_epochs']), 'patience': int(section['patience']), 'batch_size': int(section['batch_size']), 'lr': float(section['lr']), 'weight_decay': float(section['weight_decay']), 'scheduler': 'plateau', 'lr_scheduler_mode': 'min', 'lr_scheduler_monitor': 'val_hint_mse', 'lr_factor': float(section['lr_factor']), 'lr_patience': int(section['lr_patience']), 'warmup_epochs': 0}

    def baseline_task(self, stage: str, name: str, full: bool, fold: int=0, **values) -> dict:
        section = self.config['student']['baseline']
        return {**self.common(stage, name, fold), 'kind': 'baseline', 'method': 'baseline', 'ce_mode': 'supervised_ce', 'student_hidden': self.config['student']['hidden'], 'max_epochs': int(section['final_epochs'] if full else section['search_max_epochs']), 'patience': int(section['final_patience'] if full else section['search_patience']), 'batch_size': int(values['batch_size']), 'eval_batch_size': int(values['batch_size']), 'lr': float(values['lr']), 'weight_decay': float(values['weight_decay']), 'label_smoothing': float(values['label_smoothing']), 'scheduler': 'plateau', 'lr_scheduler_mode': 'max', 'lr_scheduler_monitor': 'val_accuracy', 'lr_factor': float(section['lr_factor']), 'lr_patience': int(values['lr_patience']), 'warmup_epochs': 0, 'l1': 0.0, 'grad_clip': 1.0, 'training_profile': str(values['training_profile']), **({} if full else {'train_samples': int(self.config['search']['train_samples']), 'val_samples': int(self.config['search']['val_samples'])})}

    def distill_task(self, method: str, stage: str, name: str, full: bool, fold: int=0, **values) -> dict:
        search = self.config['search']
        if method == 'vrm':
            section = self.config['student']['vrm']
            max_epochs = int(section['final_epochs'] if full else search['max_epochs'])
            patience = int(section['final_patience'] if full else search['patience'])
            warmup_fraction = float(values.pop('warmup_fraction', 0.05))
            task = {**self.common(stage, name, fold), 'method': method, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': max_epochs, 'patience': patience, 'batch_size': int(values.pop('batch_size', 256)), 'eval_batch_size': int(values.get('eval_batch_size', 512)), 'weight_decay': float(values.pop('weight_decay', 0.0)), 'warmup_epochs': int(round(max_epochs * warmup_fraction)), 'warmup_fraction': warmup_fraction, 'scheduler': 'warmup_cosine', 'max_pairs': int(self.config['distillation']['max_pairs']), 'virtual_views': False, 'pruning': False, 'l1': 0.0, 'grad_clip': 1.0, **values}
        else:
            section = self.config['student']['kd']
            max_epochs = int(section['final_epochs'] if full else search['max_epochs'])
            patience = int(section['final_patience'] if full else search['patience'])
            task = {**self.common(stage, name, fold), 'method': method, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': max_epochs, 'patience': patience, 'batch_size': int(section['batch_size']), 'eval_batch_size': int(section['batch_size']), 'weight_decay': float(section['weight_decay']), 'scheduler': 'plateau', 'lr_scheduler_mode': 'min', 'lr_scheduler_monitor': 'val_nll', 'lr_factor': float(section['lr_factor']), 'lr_patience': int(section['lr_patience']), 'warmup_epochs': 0, 'l1': 0.0, 'grad_clip': 1.0, **values}
        if not full:
            task['train_samples'] = int(search['train_samples'])
            task['val_samples'] = int(search['val_samples'])
        return task

    def job(self, task: dict, output: Path, runner: str='distill') -> dict:
        return {'runner': runner, 'task': task, 'output': output}

    def run_phase(self, name: str, jobs: list[dict]) -> None:
        for index, job in enumerate(jobs):
            fold = int(job['task'].get('fold', 0))
            if job['runner'] == 'cache':
                cache_aligned_teacher(self.args.data_dir, self.teacher_checkpoints[fold], self.cache_dir(fold), self.device, int(self.config['teacher_cache_batch_size']))
            elif job['runner'] == 'hint':
                train_hint_pretrain(job['task'], self.args.data_dir, self.cache_dir(fold), job['output'], self.device)
            elif job['runner'] == 'supervised':
                train_supervised(job['task'], self.args.data_dir, job['output'], self.device)
            elif job['runner'] == 'distill':
                checkpoint = self.hint_checkpoint(fold) if job['task']['method'] in ('hint', 'hint_beta') else None
                train_distillation(job['task'], self.args.data_dir, self.cache_dir(fold), job['output'], self.device, checkpoint)
            elif job['runner'] == 'evaluate':
                evaluate_best_student(job['checkpoint'], self.args.data_dir, self.cache_dir(fold), job['output'], self.device, self.config['evaluation'], job['selection'])
            elif job['runner'] == 'evaluate_teacher':
                evaluate_teacher(self.teacher_checkpoints[fold], self.args.data_dir, job['output'], self.device, self.config['evaluation'], fold)
            else:
                raise ValueError(f"Unknown runner: {job['runner']}")

    @staticmethod
    def results(jobs: list[dict]) -> list[dict]:
        return [json.loads((job['output'] / 'result.json').read_text(encoding='utf-8')) for job in jobs]

    def select(self, name: str, jobs: list[dict], output: Path) -> dict:
        select_best(self.results(jobs), output, name)
        return json.loads(output.read_text(encoding='utf-8'))

    def baseline_search(self) -> tuple[list[dict], dict]:
        section = self.config['student']['baseline']
        reference = {'batch_size': int(section['reference_batch_size']), 'weight_decay': 0.0, 'label_smoothing': 0.0, 'lr_patience': int(section['reference_lr_patience']), 'training_profile': 'notebook_reference'}
        lr_jobs = []
        for lr in section['learning_rates']:
            task = self.baseline_task('baseline_lr_search', f'lr_{float_tag(lr)}', False, lr=float(lr), **reference)
            lr_jobs.append(self.job(task, self.output / '03_search' / 'baseline' / 'lr' / task['name'], runner='supervised'))
        self.run_phase('baseline_lr_search', lr_jobs)
        best_lr_selection = self.select('baseline_lr', lr_jobs, self.output / '03_search' / 'baseline' / 'best_lr.json')
        best_lr = float(best_lr_selection['best']['lr'])
        reference_job = next((job for job in lr_jobs if float(job['task']['lr']) == best_lr))
        profile_jobs = []
        for profile in section['profiles']:
            task = self.baseline_task('baseline_training_and_regularization_search', str(profile['name']), False, lr=best_lr, batch_size=int(profile['batch_size']), weight_decay=float(profile['weight_decay']), label_smoothing=float(profile['label_smoothing']), lr_patience=int(profile['lr_patience']), training_profile=str(profile['name']))
            profile_jobs.append(self.job(task, self.output / '03_search' / 'baseline' / 'training_and_regularization' / task['name'], runner='supervised'))
        self.run_phase('baseline_training_and_regularization_search', profile_jobs)
        selected = self.select('baseline_training_and_regularization', [reference_job, *profile_jobs], self.output / '03_search' / 'baseline' / 'selected.json')['best']
        return (lr_jobs + profile_jobs, selected)

    def temperature_and_ce_searches(self) -> tuple[list[dict], dict]:
        distill = self.config['distillation']
        lr = float(self.config['student']['kd']['lr'])
        selected: dict[tuple[str, str], dict] = {}
        all_jobs: list[dict] = []
        temperature_jobs: dict[tuple[str, str], list[dict]] = {}
        no_ce_jobs = []
        for method in ('kd', 'hint'):
            jobs = []
            for temperature in distill['temperatures']:
                task = self.distill_task(method, f'{method}_no_ce_temperature_search', f'T_{float_tag(temperature)}', False, ce_mode='no_ce', ce_weight=0.0, temperature=float(temperature), beta=0.0, lr=lr)
                jobs.append(self.job(task, self.output / '03_search' / method / 'no_ce' / 'temperature' / task['name']))
            temperature_jobs[method, 'no_ce'] = jobs
            no_ce_jobs.extend(jobs)
            all_jobs.extend(jobs)
        self.run_phase('kd_hint_no_ce_temperature_search', no_ce_jobs)
        no_ce_best = {method: self.select(f'{method}_no_ce_temperature', temperature_jobs[method, 'no_ce'], self.output / '03_search' / method / 'no_ce' / 'best_temperature.json') for method in ('kd', 'hint')}
        ce_jobs_all = []
        ce_jobs: dict[str, list[dict]] = {}
        for method in ('kd', 'hint'):
            temperature = float(no_ce_best[method]['best']['temperature'])
            jobs = []
            for ce_weight in distill['ce_weights']:
                task = self.distill_task(method, f'{method}_ce_weight_search', f'ce_{float_tag(ce_weight)}', False, ce_mode='with_ce', ce_weight=float(ce_weight), temperature=temperature, beta=0.0, lr=lr)
                jobs.append(self.job(task, self.output / '03_search' / method / 'with_ce' / 'ce_weight' / task['name']))
            ce_jobs[method] = jobs
            ce_jobs_all.extend(jobs)
            all_jobs.extend(jobs)
        self.run_phase('kd_hint_ce_weight_search', ce_jobs_all)
        ce_best = {method: self.select(f'{method}_ce_weight', ce_jobs[method], self.output / '03_search' / method / 'with_ce' / 'best_ce_weight.json') for method in ('kd', 'hint')}
        with_ce_jobs_all = []
        for method in ('kd', 'hint'):
            ce_weight = float(ce_best[method]['best']['ce_weight'])
            jobs = []
            for temperature in distill['temperatures']:
                task = self.distill_task(method, f'{method}_with_ce_temperature_search', f'T_{float_tag(temperature)}', False, ce_mode='with_ce', ce_weight=ce_weight, temperature=float(temperature), beta=0.0, lr=lr)
                jobs.append(self.job(task, self.output / '03_search' / method / 'with_ce' / 'temperature' / task['name']))
            temperature_jobs[method, 'with_ce'] = jobs
            with_ce_jobs_all.extend(jobs)
            all_jobs.extend(jobs)
        self.run_phase('kd_hint_with_ce_temperature_search', with_ce_jobs_all)
        for method in ('kd', 'hint'):
            for mode in ('no_ce', 'with_ce'):
                best = self.select(f'{method}_{mode}_temperature', temperature_jobs[method, mode], self.output / '03_search' / method / mode / 'selected.json')['best']
                selected[method, mode] = best
        hint_with_ce_temperature = float(selected['hint', 'with_ce']['temperature'])
        hint_beta_ce_jobs = []
        for ce_weight in distill['ce_weights']:
            task = self.distill_task('hint_beta', 'hint_beta_ce_weight_search', f'ce_{float_tag(ce_weight)}', False, ce_mode='with_ce', ce_weight=float(ce_weight), temperature=hint_with_ce_temperature, beta=float(distill['hint_beta_ce_reference']), lr=lr)
            hint_beta_ce_jobs.append(self.job(task, self.output / '03_search' / 'hint_beta' / 'with_ce' / 'ce_weight' / task['name']))
        all_jobs.extend(hint_beta_ce_jobs)
        self.run_phase('hint_beta_ce_weight_search', hint_beta_ce_jobs)
        hint_beta_ce = float(self.select('hint_beta_ce_weight', hint_beta_ce_jobs, self.output / '03_search' / 'hint_beta' / 'with_ce' / 'best_ce_weight.json')['best']['ce_weight'])
        for mode in ('no_ce', 'with_ce'):
            ce_weight = 0.0 if mode == 'no_ce' else hint_beta_ce
            reference_temperature = float(selected['hint', mode]['temperature'])
            beta_jobs = []
            for beta in distill['beta_values']:
                task = self.distill_task('hint_beta', f'hint_beta_{mode}_beta_search', f'beta_{float_tag(beta)}', False, ce_mode=mode, ce_weight=ce_weight, temperature=reference_temperature, beta=float(beta), lr=lr)
                beta_jobs.append(self.job(task, self.output / '03_search' / 'hint_beta' / mode / 'beta' / task['name']))
            all_jobs.extend(beta_jobs)
            self.run_phase(f'hint_beta_{mode}_beta_search', beta_jobs)
            beta = float(self.select(f'hint_beta_{mode}_beta', beta_jobs, self.output / '03_search' / 'hint_beta' / mode / 'best_beta.json')['best']['beta'])
            jobs = []
            for temperature in distill['temperatures']:
                task = self.distill_task('hint_beta', f'hint_beta_{mode}_temperature_search', f'T_{float_tag(temperature)}', False, ce_mode=mode, ce_weight=ce_weight, temperature=float(temperature), beta=beta, lr=lr)
                jobs.append(self.job(task, self.output / '03_search' / 'hint_beta' / mode / 'temperature' / task['name']))
            all_jobs.extend(jobs)
            self.run_phase(f'hint_beta_{mode}_temperature_search', jobs)
            selected['hint_beta', mode] = self.select(f'hint_beta_{mode}_temperature', jobs, self.output / '03_search' / 'hint_beta' / mode / 'selected.json')['best']
        return (all_jobs, selected)

    def vrm_searches(self) -> tuple[list[dict], dict]:
        distill = self.config['distillation']
        reference_profile = distill['vrm_training_profiles'][1 if len(distill['vrm_training_profiles']) > 1 else 0]
        all_jobs: list[dict] = []
        relation_winners: dict[str, dict] = {}
        for mode in ('no_ce', 'with_ce'):
            criterion_winners = []
            for criterion in distill['relation_criteria']:
                if mode == 'no_ce':
                    values = [(1.0, float(ratio), f'ratio_{float_tag(ratio)}') for ratio in distill['vrm_no_ce_lambda_ic_ratios']]
                    ce_weight = 0.0
                else:
                    values = [(float(pair[0]), float(pair[1]), f'lis_{float_tag(pair[0])}_lic_{float_tag(pair[1])}') for pair in distill['vrm_with_ce_pairs']]
                    ce_weight = 1.0
                jobs = []
                for lambda_is, lambda_ic, name in values:
                    task = self.distill_task('vrm', f'vrm_{criterion}_{mode}_relation_search', name, False, ce_mode=mode, ce_weight=ce_weight, criterion=criterion, lambda_is=lambda_is, lambda_ic=lambda_ic, lr=float(distill['vrm_relation_lr']), batch_size=int(reference_profile['batch_size']), warmup_fraction=float(reference_profile['warmup_fraction']), weight_decay=float(reference_profile['weight_decay']), training_profile=reference_profile['name'])
                    jobs.append(self.job(task, self.output / '03_search' / 'vrm' / mode / criterion / 'relation' / name))
                all_jobs.extend(jobs)
                self.run_phase(f'vrm_{criterion}_{mode}_relation_search', jobs)
                criterion_winners.append(self.select(f'vrm_{criterion}_{mode}_relation', jobs, self.output / '03_search' / 'vrm' / mode / criterion / 'best_relation.json')['best'])
            global_path = self.output / '03_search' / 'vrm' / mode / 'best_criterion_and_relation.json'
            select_best(criterion_winners, global_path, f'vrm_{mode}_criterion_and_relation')
            relation_winners[mode] = json.loads(global_path.read_text(encoding='utf-8'))['best']
        lr_winners: dict[str, dict] = {}
        for mode in ('no_ce', 'with_ce'):
            chosen = relation_winners[mode]
            jobs = []
            for lr in distill['vrm_learning_rates']:
                task = self.distill_task('vrm', f'vrm_{mode}_lr_search', f'lr_{float_tag(lr)}', False, ce_mode=mode, ce_weight=0.0 if mode == 'no_ce' else 1.0, criterion=chosen['criterion'], lambda_is=float(chosen['lambda_is']), lambda_ic=float(chosen['lambda_ic']), lr=float(lr), batch_size=int(reference_profile['batch_size']), warmup_fraction=float(reference_profile['warmup_fraction']), weight_decay=float(reference_profile['weight_decay']), training_profile=reference_profile['name'])
                jobs.append(self.job(task, self.output / '03_search' / 'vrm' / mode / 'lr' / task['name']))
            all_jobs.extend(jobs)
            self.run_phase(f'vrm_{mode}_lr_search', jobs)
            lr_winners[mode] = self.select(f'vrm_{mode}_lr', jobs, self.output / '03_search' / 'vrm' / mode / 'best_lr.json')['best']
        selected: dict[str, dict] = {}
        for mode in ('no_ce', 'with_ce'):
            chosen = lr_winners[mode]
            jobs = []
            for profile in distill['vrm_training_profiles']:
                task = self.distill_task('vrm', f'vrm_{mode}_training_process_search', profile['name'], False, ce_mode=mode, ce_weight=0.0 if mode == 'no_ce' else 1.0, criterion=chosen['criterion'], lambda_is=float(chosen['lambda_is']), lambda_ic=float(chosen['lambda_ic']), lr=float(chosen['lr']), batch_size=int(profile['batch_size']), warmup_fraction=float(profile['warmup_fraction']), weight_decay=float(profile['weight_decay']), training_profile=profile['name'])
                jobs.append(self.job(task, self.output / '03_search' / 'vrm' / mode / 'training_process' / task['name']))
            all_jobs.extend(jobs)
            self.run_phase(f'vrm_{mode}_training_process_search', jobs)
            selected[mode] = self.select(f'vrm_{mode}_training_process', jobs, self.output / '03_search' / 'vrm' / mode / 'selected.json')['best']
        return (all_jobs, selected)

    def final_jobs(self, selected_baseline: dict, selected_kd: dict, selected_vrm: dict) -> list[dict]:
        jobs = []
        for fold in self.folds:
            baseline = self.baseline_task('baseline_final', 'final', True, fold=fold, lr=float(selected_baseline['lr']), batch_size=int(selected_baseline['batch_size']), weight_decay=float(selected_baseline['weight_decay']), label_smoothing=float(selected_baseline['label_smoothing']), lr_patience=int(selected_baseline['lr_patience']), training_profile=str(selected_baseline['training_profile']))
            jobs.append(self.job(baseline, self.output / '04_final' / f'fold{fold}' / 'baseline' / 'supervised_ce', runner='supervised'))
            for method in ('kd', 'hint', 'hint_beta'):
                for mode in ('no_ce', 'with_ce'):
                    chosen = selected_kd[method, mode]
                    task = self.distill_task(method, f'{method}_{mode}_final', 'final', True, fold=fold, ce_mode=mode, ce_weight=float(chosen['ce_weight']), temperature=float(chosen['temperature']), beta=float(chosen.get('beta', 0.0)), lr=float(chosen['lr']))
                    jobs.append(self.job(task, self.output / '04_final' / f'fold{fold}' / method / mode))
            for mode in ('no_ce', 'with_ce'):
                chosen = selected_vrm[mode]
                task = self.distill_task('vrm', f'vrm_{mode}_final', 'final', True, fold=fold, ce_mode=mode, ce_weight=0.0 if mode == 'no_ce' else 1.0, criterion=chosen['criterion'], lambda_is=float(chosen['lambda_is']), lambda_ic=float(chosen['lambda_ic']), lr=float(chosen['lr']), batch_size=int(chosen['batch_size']), warmup_fraction=float(chosen['warmup_fraction']), weight_decay=float(chosen['weight_decay']), training_profile=chosen['training_profile'])
                jobs.append(self.job(task, self.output / '04_final' / f'fold{fold}' / 'vrm' / mode))
        return jobs

    def select_fivefold(self, jobs: list[dict], output: Path) -> dict:
        best, models = select_fivefold_results(self.results(jobs), self.folds, 9)
        atomic_json(output, {'selection': 'best_model_fivefold', 'metric': 'five-fold mean best validation accuracy; five-fold mean validation NLL tie-break', 'dispersion_used_for_selection': False, 'best': best, 'models': models})
        return json.loads(output.read_text(encoding='utf-8'))

    def evaluation_jobs(self, selection: dict, model: dict, label: str) -> list[dict]:
        jobs = []
        for fold_result in model['fold_results']:
            fold = int(fold_result['fold'])
            task = self.common(f'{label}_evaluation', f'fold{fold}', fold)
            job = self.job(task, self.output / '05_evaluation' / label / f'fold{fold}.json', runner='evaluate')
            job['checkpoint'] = Path(fold_result['output_dir']) / 'best.pt'
            job['selection'] = {'metric': selection['metric'], 'method': model['method'], 'ce_mode': model['ce_mode'], 'fold': fold}
            jobs.append(job)
        return jobs

    def teacher_evaluation_jobs(self) -> list[dict]:
        return [self.job(self.common('teacher_evaluation', f'fold{fold}', fold), self.output / '05_evaluation' / 'teacher' / f'fold{fold}.json', runner='evaluate_teacher') for fold in self.folds]

    def candidate_evaluations(self, finals, kd_searches, vrm_searches):
        trained = list(finals)
        trained.extend(job for job in kd_searches if 'temperature' in job['task']['stage'])
        relation_outputs = set()
        for mode in ('no_ce', 'with_ce'):
            for criterion in self.config['distillation']['relation_criteria']:
                path = self.output / '03_search' / 'vrm' / mode / criterion / 'best_relation.json'
                selected = json.loads(path.read_text(encoding='utf-8'))['best']
                relation_outputs.add(str(Path(selected['output_dir'])))
        trained.extend(job for job in vrm_searches if str(job['output']) in relation_outputs)
        evaluations = []
        for job in trained:
            task = job['task']
            fold = int(task.get('fold', 0))
            metadata = {key: task.get(key) for key in ('stage', 'method', 'ce_mode', 'temperature', 'criterion', 'beta', 'ce_weight', 'lambda_is', 'lambda_ic')}
            evaluation = evaluate_best_student(job['output'] / 'best.pt', self.args.data_dir, self.cache_dir(fold), job['output'] / 'evaluation.json', self.device, self.config['evaluation'], metadata)
            evaluations.append({**metadata, **evaluation})
        atomic_json(self.output / '05_evaluation' / 'all_candidates.json', {'evaluations': evaluations})
        rows = [{key: value for key, value in evaluation.items() if key in ('stage', 'method', 'ce_mode', 'temperature', 'criterion', 'beta', 'ce_weight', 'lambda_is', 'lambda_ic', 'fold', 'checkpoint')} | evaluation['metrics'] for evaluation in evaluations]
        with (self.output / '05_evaluation' / 'all_candidates.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def aggregate_evaluations(self, evaluations: list[dict], selection: dict, model: dict, output_name: str) -> dict:
        metric_names = ('accuracy', 'nll', 'ece', 'lorentz_invariance_1_jsd', 'lorentz_invariance_top1_agreement', 'lorentz_relative_logit_error', 'fid_top1_agreement', 'fid_1_jsd')
        evaluations, aggregate_metrics = aggregate_fivefold_metrics(evaluations, self.folds, metric_names)
        folds = [int(value['fold']) for value in evaluations]
        result = {'status': 'complete', 'selection_role': 'reference_only' if model['method'] == 'teacher' else 'candidate', 'selection_metric': None if model['method'] == 'teacher' else selection['metric'], 'dispersion_used_for_selection': False, 'method': model['method'], 'ce_mode': model['ce_mode'], 'folds': folds, 'metrics': aggregate_metrics, 'fold_evaluations': evaluations}
        atomic_json(self.output / '05_evaluation' / f'{output_name}.json', result)
        csv_path = self.output / '05_evaluation' / f'{output_name}.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['row', 'fold', 'method', 'ce_mode', *metric_names]
        rows = []
        for evaluation in evaluations:
            rows.append({'row': 'fold', 'fold': int(evaluation['fold']), 'method': result['method'], 'ce_mode': result['ce_mode'], **{name: evaluation['metrics'][name] for name in metric_names}})
        rows.append({'row': 'mean', 'fold': '', 'method': result['method'], 'ce_mode': result['ce_mode'], **{name: aggregate_metrics[name]['mean'] for name in metric_names}})
        rows.append({'row': 'population_std', 'fold': '', 'method': result['method'], 'ce_mode': result['ce_mode'], **{name: aggregate_metrics[name]['population_std'] for name in metric_names}})
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return result

    def summarize(self, teachers: list[dict], searches: list[dict], finals: list[dict], selection: dict, evaluation: dict, baseline_evaluation: dict, teacher_evaluation: dict) -> None:
        search_results = self.results(searches)
        final_results = self.results(finals)
        write_results_csv(search_results, self.output / '05_summary' / 'search_results.csv')
        write_results_csv(final_results, self.output / '05_summary' / 'final_results.csv')
        atomic_json(self.output / '05_summary' / 'search_results.json', {'results': search_results})
        atomic_json(self.output / '05_summary' / 'final_results.json', {'results': final_results})
        summary = {'status': 'complete', 'teachers': teachers, 'folds': self.folds, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': self.config['student']['guided_hidden_index'], 'notebook_guided_module_index': self.config['student']['notebook_guided_module_index'], 'search_trials': len(search_results), 'full_training_models': len(final_results), 'selection_metric': selection['metric'], 'dispersion_used_for_selection': False, 'best_method': selection['best'], 'best_method_evaluation': evaluation, 'baseline_evaluation': baseline_evaluation, 'teacher_evaluation': teacher_evaluation, 'temperature_scan': self.config['distillation']['temperatures'], 'baseline': {'method': 'supervised cross-entropy', 'fold0_learning_rate_scan': self.config['student']['baseline']['learning_rates'], 'fold0_training_and_regularization_profiles': self.config['student']['baseline']['profiles']}, 'ce_modes': ['no_ce', 'with_ce'], 'relation_criteria': self.config['distillation']['relation_criteria'], 'vrm_virtual_views': False, 'vrm_pruning': False, 'compute_precision': self.config['compute_precision']}
        atomic_json(self.output / 'summary.json', summary)

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = verify_aligned_dataset(self.args.data_dir)
        teachers = [{'fold': fold, 'checkpoint': str(self.teacher_checkpoints[fold]), 'config': self.teacher_configs[fold]} for fold in self.folds]
        atomic_json(self.output / '00_data' / 'alignment_manifest.json', manifest)
        atomic_json(self.output / '00_data' / 'teachers.json', {'teachers': teachers})
        teachers = json.loads((self.output / '00_data' / 'teachers.json').read_text(encoding='utf-8'))['teachers']
        cache_jobs = [self.job(self.common('teacher_cache', f'fold{fold}', fold), self.cache_dir(fold), runner='cache') for fold in self.folds]
        self.run_phase('teacher_cache_fivefold', cache_jobs)
        hint_jobs = [self.job(self.hint_task(fold), self.output / '02_hint_pretrain' / f'fold{fold}', runner='hint') for fold in self.folds]
        self.run_phase('hint_pretrain_fivefold', hint_jobs)
        baseline_searches, selected_baseline = self.baseline_search()
        kd_searches, selected_kd = self.temperature_and_ce_searches()
        vrm_searches, selected_vrm = self.vrm_searches()
        finals = self.final_jobs(selected_baseline, selected_kd, selected_vrm)
        self.run_phase('final_full_training_fivefold', finals)
        selection = self.select_fivefold(finals, self.output / '04_final' / 'fivefold_selection.json')
        best_model = selection['best']
        baseline_model = next((model for model in selection['models'] if model['method'] == 'baseline'))
        baseline_jobs = self.evaluation_jobs(selection, baseline_model, 'baseline')
        teacher_jobs = self.teacher_evaluation_jobs()
        if best_model['method'] == 'baseline':
            best_jobs = baseline_jobs
            all_evaluation_jobs = baseline_jobs + teacher_jobs
        else:
            best_jobs = self.evaluation_jobs(selection, best_model, 'best')
            all_evaluation_jobs = best_jobs + baseline_jobs + teacher_jobs
        self.run_phase('best_baseline_teacher_full_metrics_fivefold', all_evaluation_jobs)
        best_evaluations = [json.loads(job['output'].read_text(encoding='utf-8')) for job in best_jobs]
        baseline_evaluations = [json.loads(job['output'].read_text(encoding='utf-8')) for job in baseline_jobs]
        teacher_evaluations = [json.loads(job['output'].read_text(encoding='utf-8')) for job in teacher_jobs]
        evaluation = self.aggregate_evaluations(best_evaluations, selection, best_model, 'fivefold')
        baseline_evaluation = self.aggregate_evaluations(baseline_evaluations, selection, baseline_model, 'baseline_fivefold')
        teacher_model = {'method': 'teacher', 'ce_mode': 'not_applicable'}
        teacher_evaluation = self.aggregate_evaluations(teacher_evaluations, selection, teacher_model, 'teacher_fivefold')
        self.candidate_evaluations(finals, kd_searches, vrm_searches)
        self.summarize(teachers, baseline_searches + kd_searches + vrm_searches, finals, selection, evaluation, baseline_evaluation, teacher_evaluation)
