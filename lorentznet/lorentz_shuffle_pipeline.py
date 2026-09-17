from types import SimpleNamespace
import csv
import json
import statistics
from pathlib import Path
import torch
from .distill.advanced_training import cache_aligned_teacher, select_best, train_distillation, train_hint_pretrain, train_supervised, write_results_csv
from .distill.lorentz_shuffle import atomic_json, build_shuffled_dataset
from .distill.evaluation import evaluate_checkpoint

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
        self.hint_checkpoint = self.output / '04_distillation' / 'hint_pretrain' / 'best.pt'
        self.teacher_checkpoint = self.output / '01_teacher' / 'final' / 'best.pt'
        self.cache_dir = self.output / '03_teacher_cache'

    def common(self, stage: str, name: str, seed_offset: int=0) -> dict:
        return {'stage': stage, 'name': name, 'fold': int(self.config['fold']), 'n_folds': int(self.config['n_folds']), 'fold_seed': int(self.config['fold_seed']), 'seed': int(self.config['seed']) + seed_offset, 'compute_precision': 'float32_no_tf32'}

    def run_phase(self, name: str, jobs: list[dict]) -> None:
        for index, job in enumerate(jobs):
            runner = job['runner']
            task = job['task']
            output_dir = job['output']
            if runner == 'supervised':
                train_supervised(task, self.args.shuffled_data_dir, output_dir, self.device)
            elif runner == 'hint':
                train_hint_pretrain(task, self.args.shuffled_data_dir, self.cache_dir, output_dir, self.device)
            elif runner == 'distill':
                hint_checkpoint = self.hint_checkpoint if task['method'] in ('hint', 'hint_beta') else None
                train_distillation(task, self.args.shuffled_data_dir, self.cache_dir, output_dir, self.device, hint_checkpoint)
            elif runner == 'evaluate':
                evaluate_checkpoint(job['checkpoint'], job['model_kind'], self.args.shuffled_data_dir, self.cache_dir, output_dir / 'evaluation.json', self.device, self.config['evaluation'], task)
            else:
                raise ValueError(f'Unknown runner: {runner}')

    @staticmethod
    def results(jobs: list[dict]) -> list[dict]:
        return [json.loads((job['output'] / 'result.json').read_text(encoding='utf-8')) for job in jobs]

    def select(self, name: str, jobs: list[dict], path: Path) -> dict:
        select_best(self.results(jobs), path, name)
        return json.loads(path.read_text(encoding='utf-8'))

    def teacher_task(self, values: dict, final: bool, index: int) -> dict:
        teacher = self.config['teacher']
        task = {**self.common('teacher_final' if final else 'teacher_search', values['name'], index), 'kind': 'teacher', **values, 'max_epochs': int(teacher['final_epochs'] if final else teacher['search_epochs']), 'patience': int(teacher['final_patience'] if final else teacher['search_patience']), 'grad_clip': 1.0, 'bn_calibration_samples': int(teacher['bn_calibration_samples']), 'eval_batch_size': int(values['batch_size'])}
        if not final:
            task['train_samples'] = int(teacher['search_train_samples'])
            task['val_samples'] = int(teacher['search_val_samples'])
        return task

    def baseline_task(self) -> dict:
        values = self.config['student']['baseline']
        return {**self.common('mlp_baseline', 'notebook_mlp'), 'kind': 'baseline', 'student_hidden': self.config['student']['hidden'], 'max_epochs': int(values['max_epochs']), 'lr': float(values['lr']), 'weight_decay': float(values['weight_decay']), 'batch_size': int(values['batch_size']), 'eval_batch_size': int(values['batch_size']), 'lr_factor': float(values['lr_factor']), 'lr_patience': int(values['lr_patience']), 'lr_scheduler_mode': values['lr_scheduler_mode'], 'lr_scheduler_monitor': values['lr_scheduler_monitor'], 'patience': int(values['patience']), 'label_smoothing': 0.0, 'l1': 0.0, 'grad_clip': 1.0, 'notebook_reference': "colab_kd_notebook/build_notebook.py CFG['baseline']"}

    def distill_task(self, method: str, stage: str, name: str, index: int, full: bool, **values) -> dict:
        section_name = 'vrm' if method == 'vrm' else 'kd'
        section = self.config['student'][section_name]
        search = self.config['distillation']
        max_epochs = int(section['max_epochs'] if full else search['search_epochs'])
        patience = int(section['patience'] if full else search['search_patience'])
        task = {**self.common(stage, name, 1000 + index), 'method': method, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': max_epochs, 'patience': patience, 'batch_size': int(section['batch_size']), 'eval_batch_size': int(section['batch_size']), 'weight_decay': float(section['weight_decay']), 'l1': 0.0, 'grad_clip': 1.0, **values}
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
        if not full:
            task['train_samples'] = int(search['search_train_samples'])
            task['val_samples'] = int(search['search_val_samples'])
        return task

    def hint_task(self) -> dict:
        section = self.config['student']['hint']
        return {**self.common('hint_pretrain', 'notebook_hint'), 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': int(self.config['student']['guided_hidden_index']), 'max_epochs': int(section['max_epochs']), 'patience': int(section['patience']), 'batch_size': int(section['batch_size']), 'lr': float(section['lr']), 'weight_decay': float(section['weight_decay']), 'warmup_epochs': 0, 'scheduler': 'plateau', 'lr_factor': float(section['lr_factor']), 'lr_patience': int(section['lr_patience']), 'lr_scheduler_mode': section['lr_scheduler_mode'], 'lr_scheduler_monitor': section['lr_scheduler_monitor'], 'notebook_reference': "colab_kd_notebook/build_notebook.py CFG['hint']"}

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = build_shuffled_dataset(self.args.source_data_dir, self.args.shuffled_data_dir, int(self.config['dataset']['seed']), int(self.config['dataset']['chunk_size']), None)
        data_reference = {'dataset_directory': str(self.args.shuffled_data_dir), 'samples': manifest['samples'], 'seed': manifest['seed'], 'reference_lorentz_source': manifest['reference_lorentz_source'], 'reference_decomposition': manifest['reference_decomposition'], 'reference_rapidity_distribution': manifest['reference_rapidity_distribution']}
        atomic_json(self.output / '00_data' / 'dataset_reference.json', data_reference)
        teacher_jobs = []
        for index, trial in enumerate(self.config['teacher']['trials']):
            teacher_jobs.append({'runner': 'supervised', 'task': self.teacher_task(trial, False, index), 'output': self.output / '01_teacher' / 'search' / trial['name']})
        self.run_phase('teacher_search', teacher_jobs)
        teacher_selection_path = self.output / '01_teacher' / 'selection.json'
        selection = self.select('teacher_local_parameter_search', teacher_jobs, teacher_selection_path)
        best_values = {key: selection['best'][key] for key in ('name', 'hidden_dim', 'layers', 'lr', 'c_weight', 'batch_size', 'dropout', 'weight_decay', 'label_smoothing', 'warmup_epochs')}
        best_values['name'] = 'best_lorentz_shuffled'
        final_teacher_job = {'runner': 'supervised', 'task': self.teacher_task(best_values, True, 0), 'output': self.output / '01_teacher' / 'final'}
        self.run_phase('teacher_final', [final_teacher_job])
        teacher_result_path = self.output / '01_teacher' / 'training_result.json'
        result = self.results([final_teacher_job])[0]
        teacher_result = {'observed_best_validation_accuracy': float(result['best_val_accuracy']), 'observed_best_validation_nll': float(result['best_val_nll']), 'teacher_result': str(final_teacher_job['output'] / 'result.json')}
        atomic_json(teacher_result_path, teacher_result)
        baseline_job = {'runner': 'supervised', 'task': self.baseline_task(), 'output': self.output / '02_mlp_baseline'}
        self.run_phase('mlp_baseline', [baseline_job])
        cache_aligned_teacher(self.args.shuffled_data_dir, self.teacher_checkpoint, self.cache_dir, self.device, int(self.config['teacher_cache_batch_size']))
        hint_job = {'runner': 'hint', 'task': self.hint_task(), 'output': self.output / '04_distillation' / 'hint_pretrain'}
        self.run_phase('hint_pretrain', [hint_job])
        distill = self.config['distillation']
        kd_lr = float(self.config['student']['kd']['lr'])
        temperature_jobs: dict[tuple[str, str], list[dict]] = {}
        for method in ('kd', 'hint'):
            jobs = []
            for index, temperature in enumerate(distill['temperatures']):
                task = self.distill_task(method, f'{method}_no_ce_temperature', f'T_{float_tag(temperature)}', index, True, ce_mode='no_ce', ce_weight=0.0, temperature=float(temperature), beta=0.0, lr=kd_lr)
                jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / method / 'no_ce' / 'temperature' / task['name']})
            temperature_jobs[method, 'no_ce'] = jobs
        no_ce_jobs = temperature_jobs['kd', 'no_ce'] + temperature_jobs['hint', 'no_ce']
        self.run_phase('kd_hint_no_ce_temperature', no_ce_jobs)
        no_ce_best = {}
        for method in ('kd', 'hint'):
            no_ce_best[method] = self.select(f'{method}_no_ce_best_temperature', temperature_jobs[method, 'no_ce'], self.output / '04_distillation' / method / 'no_ce' / 'best_temperature.json')
        ce_search_jobs: dict[str, list[dict]] = {}
        for method in ('kd', 'hint'):
            best_temperature = float(no_ce_best[method]['best']['temperature'])
            jobs = []
            for index, ce_weight in enumerate(distill['ce_weights']):
                task = self.distill_task(method, f'{method}_ce_search', f'ce_{float_tag(ce_weight)}', index, False, ce_mode='with_ce', ce_weight=float(ce_weight), temperature=best_temperature, beta=0.0, lr=kd_lr)
                jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / method / 'with_ce' / 'ce_search' / task['name']})
            ce_search_jobs[method] = jobs
        self.run_phase('kd_hint_ce_search', ce_search_jobs['kd'] + ce_search_jobs['hint'])
        ce_best = {}
        for method in ('kd', 'hint'):
            ce_best[method] = self.select(f'{method}_best_ce_weight', ce_search_jobs[method], self.output / '04_distillation' / method / 'with_ce' / 'best_ce_weight.json')
        for method in ('kd', 'hint'):
            jobs = []
            chosen_ce = float(ce_best[method]['best']['ce_weight'])
            for index, temperature in enumerate(distill['temperatures']):
                task = self.distill_task(method, f'{method}_with_ce_temperature', f'T_{float_tag(temperature)}', index, True, ce_mode='with_ce', ce_weight=chosen_ce, temperature=float(temperature), beta=0.0, lr=kd_lr)
                jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / method / 'with_ce' / 'temperature' / task['name']})
            temperature_jobs[method, 'with_ce'] = jobs
        with_ce_jobs = temperature_jobs['kd', 'with_ce'] + temperature_jobs['hint', 'with_ce']
        self.run_phase('kd_hint_with_ce_temperature', with_ce_jobs)
        with_ce_best = {}
        for method in ('kd', 'hint'):
            with_ce_best[method] = self.select(f'{method}_with_ce_best_temperature', temperature_jobs[method, 'with_ce'], self.output / '04_distillation' / method / 'with_ce' / 'best_temperature.json')
        hint_beta_ce_jobs = []
        hint_with_ce_temperature = float(with_ce_best['hint']['best']['temperature'])
        for index, ce_weight in enumerate(distill['ce_weights']):
            task = self.distill_task('hint_beta', 'hint_beta_ce_search', f'ce_{float_tag(ce_weight)}', index, False, ce_mode='with_ce', ce_weight=float(ce_weight), temperature=hint_with_ce_temperature, beta=0.25, lr=kd_lr)
            hint_beta_ce_jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'hint_beta' / 'with_ce' / 'ce_search' / task['name']})
        self.run_phase('hint_beta_ce_search', hint_beta_ce_jobs)
        hint_beta_ce_best = self.select('hint_beta_best_ce_weight', hint_beta_ce_jobs, self.output / '04_distillation' / 'hint_beta' / 'with_ce' / 'best_ce_weight.json')
        beta_best = {}
        beta_temperature_jobs = {}
        for mode in ('no_ce', 'with_ce'):
            if mode == 'no_ce':
                ce_weight = 0.0
                reference_temperature = float(no_ce_best['hint']['best']['temperature'])
            else:
                ce_weight = float(hint_beta_ce_best['best']['ce_weight'])
                reference_temperature = hint_with_ce_temperature
            jobs = []
            for index, beta in enumerate(distill['beta_values']):
                task = self.distill_task('hint_beta', f'hint_beta_{mode}_beta_search', f'beta_{float_tag(beta)}', index, False, ce_mode=mode, ce_weight=ce_weight, temperature=reference_temperature, beta=float(beta), lr=kd_lr)
                jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'hint_beta' / mode / 'beta_search' / task['name']})
            self.run_phase(f'hint_beta_{mode}_beta_search', jobs)
            beta_best[mode] = self.select(f'hint_beta_{mode}_best_beta', jobs, self.output / '04_distillation' / 'hint_beta' / mode / 'best_beta.json')
            chosen_beta = float(beta_best[mode]['best']['beta'])
            full_jobs = []
            for index, temperature in enumerate(distill['temperatures']):
                task = self.distill_task('hint_beta', f'hint_beta_{mode}_temperature', f'T_{float_tag(temperature)}', index, True, ce_mode=mode, ce_weight=ce_weight, temperature=float(temperature), beta=chosen_beta, lr=kd_lr)
                full_jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'hint_beta' / mode / 'temperature' / task['name']})
            self.run_phase(f'hint_beta_{mode}_temperature', full_jobs)
            beta_temperature_jobs[mode] = full_jobs
            self.select(f'hint_beta_{mode}_best_temperature', full_jobs, self.output / '04_distillation' / 'hint_beta' / mode / 'best_temperature.json')
        vrm_parameter_best = {}
        for criterion in distill['relation_criteria']:
            for mode in ('no_ce', 'with_ce'):
                jobs = []
                if mode == 'no_ce':
                    values = [(1.0, float(ratio), f'ratio_{float_tag(ratio)}') for ratio in distill['vrm_no_ce_lambda_ic_ratios']]
                    ce_weight = 0.0
                else:
                    values = [(float(lambda_is), float(lambda_ic), f'lis_{float_tag(lambda_is)}_lic_{float_tag(lambda_ic)}') for lambda_is in distill['vrm_with_ce_lambda_is'] for lambda_ic in distill['vrm_with_ce_lambda_ic']]
                    ce_weight = 1.0
                for index, (lambda_is, lambda_ic, name) in enumerate(values):
                    task = self.distill_task('vrm', f'vrm_{criterion}_{mode}_relation_search', name, index, False, ce_mode=mode, ce_weight=ce_weight, criterion=criterion, lambda_is=lambda_is, lambda_ic=lambda_ic, lr=0.003)
                    jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'vrm' / criterion / mode / 'relation_search' / name})
                self.run_phase(f'vrm_{criterion}_{mode}_relation_search', jobs)
                vrm_parameter_best[criterion, mode] = self.select(f'vrm_{criterion}_{mode}_best_relation_parameters', jobs, self.output / '04_distillation' / 'vrm' / criterion / mode / 'best_relation_parameters.json')
        vrm_lr_best = {}
        for criterion in distill['relation_criteria']:
            for mode in ('no_ce', 'with_ce'):
                chosen = vrm_parameter_best[criterion, mode]['best']
                jobs = []
                for index, lr in enumerate(distill['vrm_learning_rates']):
                    task = self.distill_task('vrm', f'vrm_{criterion}_{mode}_lr_search', f'lr_{float_tag(lr)}', index, False, ce_mode=mode, ce_weight=0.0 if mode == 'no_ce' else 1.0, criterion=criterion, lambda_is=float(chosen['lambda_is']), lambda_ic=float(chosen['lambda_ic']), lr=float(lr))
                    jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'vrm' / criterion / mode / 'lr_search' / task['name']})
                self.run_phase(f'vrm_{criterion}_{mode}_lr_search', jobs)
                vrm_lr_best[criterion, mode] = self.select(f'vrm_{criterion}_{mode}_best_lr', jobs, self.output / '04_distillation' / 'vrm' / criterion / mode / 'best_lr.json')
        vrm_final_jobs = []
        for index, (criterion, mode) in enumerate(((criterion, mode) for criterion in distill['relation_criteria'] for mode in ('no_ce', 'with_ce'))):
            relation = vrm_parameter_best[criterion, mode]['best']
            lr = float(vrm_lr_best[criterion, mode]['best']['lr'])
            task = self.distill_task('vrm', f'vrm_{criterion}_{mode}_final', 'final', index, True, ce_mode=mode, ce_weight=0.0 if mode == 'no_ce' else 1.0, criterion=criterion, lambda_is=float(relation['lambda_is']), lambda_ic=float(relation['lambda_ic']), lr=lr)
            vrm_final_jobs.append({'runner': 'distill', 'task': task, 'output': self.output / '04_distillation' / 'vrm' / criterion / mode / 'final'})
        self.run_phase('vrm_final', vrm_final_jobs)
        evaluation_jobs = []

        def add_evaluation(training_job: dict, output: Path, model_kind: str, category: str) -> None:
            training_task = training_job['task']
            metadata = {'category': category, 'stage': training_task['stage'], 'name': training_task['name'], 'method': training_task.get('method', training_task.get('kind')), 'ce_mode': training_task.get('ce_mode'), 'temperature': training_task.get('temperature'), 'criterion': training_task.get('criterion'), 'beta': training_task.get('beta'), 'ce_weight': training_task.get('ce_weight'), 'lambda_is': training_task.get('lambda_is'), 'lambda_ic': training_task.get('lambda_ic'), 'training_output': str(training_job['output'])}
            evaluation_jobs.append({'runner': 'evaluate', 'task': metadata, 'model_kind': model_kind, 'checkpoint': training_job['output'] / 'best.pt', 'output': output})
        add_evaluation(final_teacher_job, self.output / '05_evaluation' / 'reference' / 'teacher', 'teacher', 'reference')
        add_evaluation(baseline_job, self.output / '05_evaluation' / 'reference' / 'mlp_baseline', 'student', 'reference')
        for (method, mode), jobs in temperature_jobs.items():
            for job in jobs:
                add_evaluation(job, self.output / '05_evaluation' / 'temperature' / method / mode / job['task']['name'], 'student', 'temperature')
        for mode, jobs in beta_temperature_jobs.items():
            for job in jobs:
                add_evaluation(job, self.output / '05_evaluation' / 'temperature' / 'hint_beta' / mode / job['task']['name'], 'student', 'temperature')
        for job in vrm_final_jobs:
            task = job['task']
            add_evaluation(job, self.output / '05_evaluation' / 'vrm_final' / task['criterion'] / task['ce_mode'], 'student', 'vrm_final')
        self.run_phase('full_notebook_metrics', evaluation_jobs)
        teacher_result = json.loads(teacher_result_path.read_text(encoding='utf-8'))
        self.summarize(teacher_result, evaluation_jobs)

    def summarize(self, teacher_result: dict, evaluation_jobs: list[dict]) -> None:
        results = []
        for path in self.output.rglob('result.json'):
            result = json.loads(path.read_text(encoding='utf-8'))
            result['result_file'] = str(path)
            results.append(result)
        results.sort(key=lambda item: item['result_file'])
        summary_dir = self.output / '05_summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(summary_dir / 'all_results.json', {'results': results})
        write_results_csv(results, summary_dir / 'all_results.csv')
        evaluations = [json.loads((job['output'] / 'evaluation.json').read_text(encoding='utf-8')) for job in evaluation_jobs]
        expected_temperature_evaluations = 6 * len(self.config['distillation']['temperatures'])
        observed_temperature_evaluations = sum((value['model'].get('category') == 'temperature' for value in evaluations))
        if observed_temperature_evaluations != expected_temperature_evaluations:
            raise RuntimeError(f'Incomplete temperature evaluation matrix: expected {expected_temperature_evaluations}, got {observed_temperature_evaluations}')
        atomic_json(summary_dir / 'all_evaluations.json', {'evaluations': evaluations})
        evaluation_columns = ('category', 'stage', 'method', 'ce_mode', 'temperature', 'criterion', 'beta', 'ce_weight', 'lambda_is', 'lambda_ic', 'accu', 'nlll', 'ecel', 'li_jsd', 'li_agree', 'lorentz_relative_logit_error', 'top1_agreement', 'teach_stu_jsd', 'evaluation_file')
        with (summary_dir / 'all_evaluations.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=evaluation_columns)
            writer.writeheader()
            for job, evaluation in zip(evaluation_jobs, evaluations):
                row = {**evaluation['model'], **evaluation['metrics'], 'evaluation_file': str(job['output'] / 'evaluation.json')}
                writer.writerow({key: row.get(key) for key in evaluation_columns})
        distilled = [value for value in evaluations if value['model'].get('category') in ('temperature', 'vrm_final')]
        best_distilled = sorted(distilled, key=lambda value: (-float(value['metrics']['accu']), float(value['metrics']['nlll'])))[0]
        atomic_json(summary_dir / 'best_method_evaluation.json', {'selection_metric': 'full-fold accuracy; NLL tie-break', 'best': best_distilled})
        final_results = [result for result in results if result.get('stage') in ('teacher_final', 'mlp_baseline') or str(result.get('stage', '')).endswith('_final') or '_temperature' in str(result.get('stage', ''))]
        accuracies = [float(result['best_val_accuracy']) for result in final_results if 'best_val_accuracy' in result]
        summary = {'status': 'complete', 'teacher_training_result': teacher_result, 'student_hidden': self.config['student']['hidden'], 'guided_hidden_index': self.config['student']['guided_hidden_index'], 'temperatures': self.config['distillation']['temperatures'], 'relation_criteria': self.config['distillation']['relation_criteria'], 'ce_modes': ['no_ce', 'with_ce'], 'completed_trials': len(results), 'reported_models': len(final_results), 'full_metric_evaluations': len(evaluations), 'temperature_evaluations': observed_temperature_evaluations, 'expected_temperature_evaluations': expected_temperature_evaluations, 'evaluation_metrics': ['accu', 'nlll', 'ecel', 'li_jsd', 'li_agree', 'lorentz_relative_logit_error', 'top1_agreement', 'teach_stu_jsd'], 'invariance_transform_source': evaluations[0]['invariance']['transform_source'], 'best_distilled_method': best_distilled, 'reported_accuracy_mean': statistics.fmean(accuracies) if accuracies else None, 'results_json': str(summary_dir / 'all_results.json'), 'results_csv': str(summary_dir / 'all_results.csv'), 'evaluations_json': str(summary_dir / 'all_evaluations.json'), 'evaluations_csv': str(summary_dir / 'all_evaluations.csv'), 'best_method_evaluation': str(summary_dir / 'best_method_evaluation.json')}
        atomic_json(self.output / 'summary.json', summary)
