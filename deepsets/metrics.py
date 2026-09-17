import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


METRICS = ('accu', 'nlll', 'ecel', 'top1_agreement', 'teach_stu_jsd', 'pi_agree', 'pi_jsd')


def _jsd(p, q):
    middle = ((p + q) * 0.5).clamp_min(torch.finfo(p.dtype).tiny)
    return ((torch.xlogy(p, p / middle).sum(-1) + torch.xlogy(q, q / middle).sum(-1)) / (2 * math.log(2))).clamp(0, 1)


@torch.no_grad()
def _perm_inv_metrics(model, x, num_perms=10, generator=None):
    p = F.softmax(model(x), dim=1)
    original = p.argmax(1)
    jsd_sum = torch.zeros(len(x), device=x.device)
    agree_sum = torch.zeros_like(jsd_sum)
    for _ in range(num_perms):
        permutation = torch.randperm(x.size(1), generator=generator).to(x.device)
        q = F.softmax(model(x[:, permutation]), dim=1)
        jsd_sum += _jsd(p, q)
        agree_sum += q.argmax(1).eq(original)
    return 1 - jsd_sum / num_perms, agree_sum / num_perms


@torch.no_grad()
def validate_full(model, loader, device, num_perms=10, seed=42):
    model.to(device).eval()
    nll = correct = count = agree = similarity = 0
    bin_conf = torch.zeros(15, dtype=torch.float64, device=device)
    bin_correct = torch.zeros_like(bin_conf)
    generator = torch.Generator().manual_seed(seed)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        targets = y.argmax(1) if y.ndim == 2 else y.long()
        logits = model(x).float()
        probability = logits.softmax(1)
        confidence, prediction = probability.max(1)
        bins = (confidence * 15).long().clamp(max=14)
        bin_conf.scatter_add_(0, bins, confidence.double())
        bin_correct.scatter_add_(0, bins, prediction.eq(targets).double())
        correct += int(prediction.eq(targets).sum())
        nll += float(F.cross_entropy(logits, targets, reduction='sum'))
        count += len(y)
        js, ag = _perm_inv_metrics(model, x, num_perms, generator)
        agree += float(ag.sum())
        similarity += float(js.sum())
    return dict(accu=correct / count, nlll=nll / count, ecel=float((bin_correct - bin_conf).abs().sum()) / count,
                pi_jsd=similarity / count, pi_agree=agree / count)


@torch.no_grad()
def compute_fidelity(student, teacher, loader, device):
    student.to(device).eval()
    teacher.to(device).eval()
    agrees = jsds = count = 0
    for x, _ in loader:
        x = x.to(device)
        s, t = student(x), teacher(x)
        agrees += int(s.argmax(1).eq(t.argmax(1)).sum())
        jsds += float(_jsd(s.softmax(1), t.softmax(1)).sum())
        count += len(x)
    return dict(top1_agreement=agrees / count, teach_stu_jsd=1 - jsds / count)


def save_results(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    groups = {}
    for row in records:
        key = (row['regime'], row['method'], row.get('temperature'), row.get('alpha'), row.get('beta'))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        item = dict(zip(('regime', 'method', 'temperature', 'alpha', 'beta'), key))
        item['folds'] = len(rows)
        item['metrics'] = {}
        for name in METRICS:
            values = [r['metrics'].get(name) for r in rows if r['metrics'].get(name) is not None]
            item['metrics'][name] = dict(mean=float(np.mean(values)) if values else None,
                                         std=float(np.std(values)) if values else None)
        summary.append(item)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    best = {}
    for row in summary:
        key = (row['regime'], row['method'], row['alpha'], row['beta'])
        if key not in best or row['metrics']['accu']['mean'] > best[key]['metrics']['accu']['mean']:
            best[key] = row
    (output / 'best_temperature.json').write_text(json.dumps(list(best.values()), indent=2, allow_nan=False), encoding='utf-8')
    return summary
