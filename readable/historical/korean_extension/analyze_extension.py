#!/usr/bin/env python3
"""Analyze all prespecified rows, refusing incomplete/duplicate task outputs."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from run_extension import ROOT, read_jsonl, validate_package


def load(out, task, system, serial, expected):
    rows = read_jsonl(out / f'{task}_{system}_{serial}.jsonl')
    d = {r['key']: r for r in rows}
    if len(d) != len(rows) or set(d) != set(expected):
        raise ValueError(
            f'Incomplete or duplicated output: {task}/{system}/{serial}; {len(d)} of {len(expected)}'
        )
    ordered = []
    for key, source in expected.items():
        r = d[key]
        if r['system'] != system or r['serialization'] != serial:
            raise ValueError('Wrong output condition')
        if r['prompt_sha256'] != hashlib.sha256(source['prompt'].encode()).hexdigest():
            raise ValueError('Prompt mismatch')
        for k in ['gold', 'orientation', 'semantic_pair_id', 'human', 'lang']:
            if k in source and r.get(k) != source[k]:
                raise ValueError(f'Changed reference metadata {k}')
        ordered.append(r)
    return ordered


def stats(y, s):
    e = s - y
    rank = lambda fn: (
        float(fn(y, s).statistic)
        if len(y) > 1 and len(set(y)) > 1 and len(set(s)) > 1
        else None
    )
    return dict(
        n=len(y),
        spearman=rank(spearmanr),
        kendall=rank(kendalltau),
        pearson=rank(pearsonr),
        mae=float(abs(e).mean()),
        rmse=float(np.sqrt((e**2).mean())),
        bias=float(e.mean()),
        extreme_over=float((e >= 50).mean()),
        extreme_under=float((e <= -50).mean()),
    )


def transfer(out):
    source = read_jsonl(ROOT / 'data/nonkorean880.jsonl')
    expected = {r['key']: r for r in source}
    rows = {
        s: load(out, 'transfer', s, 'native', expected) for s in ['base', 'adapter']
    }
    scores = {
        s: np.array(
            [r['score'] if r['score'] is not None else np.nan for r in rows[s]],
            dtype=float,
        )
        for s in rows
    }

    if any(
        a['serialized_ids_sha256'] != b['serialized_ids_sha256']
        for a, b in zip(rows['base'], rows['adapter'])
    ):
        raise ValueError('Base/adapter input token IDs differ')
    y = np.array([r['human'] for r in source])
    langs = np.array([r['lang'] for r in source])
    paired = np.isfinite(scores['base']) & np.isfinite(scores['adapter'])
    n = int(paired.sum())
    if n < 2:
        raise ValueError(
            'Too few paired numeric outputs; no primary correlation can be computed'
        )
    result = {
        'primary': 'adapter minus base pooled Spearman on shared numeric rows',
        'expected_n': len(source),
        'paired_n': n,
        'paired_exclusions': [
            source[i]['key'] for i in range(len(source)) if not paired[i]
        ],
        'paired_languages': len(set(langs[paired])),
        'availability': {},
        'paired_metrics': {},
        'sensitivity_all_available': {},
        'per_language': [],
        'resource_groups': [],
    }
    for s in rows:
        valid = np.isfinite(scores[s])
        result['availability'][s] = {
            'numeric': int(valid.sum()),
            'unavailable': int((~valid).sum()),
            'parser_anchor': sum(r['parser_anchor'] for r in rows[s]),
            'recovered_numeric': sum(r['recovered'] for r in rows[s]),
            'at_output_limit': sum(r['reached_token_limit'] for r in rows[s]),
        }
        result['paired_metrics'][s] = stats(y[paired], scores[s][paired])
        result['sensitivity_all_available'][s] = (
            stats(y[valid], scores[s][valid]) if valid.any() else None
        )
    for lang in sorted(set(langs)):
        m = paired & (langs == lang)
        result['per_language'].append(
            {
                'lang': lang,
                'expected': int((langs == lang).sum()),
                'paired_n': int(m.sum()),
                **{s: stats(y[m], scores[s][m]) if m.any() else None for s in rows},
            }
        )
    for group in ['high', 'medium', 'low']:
        m = paired & np.array([r['resource_level'] == group for r in source])
        result['resource_groups'].append(
            {
                'group': group,
                'paired_n': int(m.sum()),
                **{s: stats(y[m], scores[s][m]) if m.any() else None for s in rows},
            }
        )
    keys = ['spearman', 'mae', 'rmse', 'bias', 'extreme_over', 'extreme_under']
    result['observed_delta_adapter_minus_base'] = {
        k: (
            (
                result['paired_metrics']['adapter'][k]
                - result['paired_metrics']['base'][k]
            )
            if result['paired_metrics']['adapter'][k] is not None
            and result['paired_metrics']['base'][k] is not None
            else None
        )
        for k in keys
    }
    groups = [np.where(paired & (langs == lang))[0] for lang in sorted(set(langs))]
    rng = np.random.default_rng(2026)
    draws = {k: [] for k in keys}
    degenerate = 0
    while len(draws['spearman']) < 2000:
        ids = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        a = stats(y[ids], scores['base'][ids])
        b = stats(y[ids], scores['adapter'][ids])
        if a['spearman'] is None or b['spearman'] is None:
            degenerate += 1
            if degenerate > 10000:
                raise ValueError(
                    'Too many degenerate bootstrap draws; report no interval'
                )
            continue
        for k in keys:
            draws[k].append(b[k] - a[k])
    result['paired_language_cluster_bootstrap'] = {
        'seed': 2026,
        'replicates': 2000,
        'degenerate_redraws': degenerate,
        'language_clusters': len(groups),
        'percentile_95_intervals': {
            k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in draws.items()
        },
        'secondary_intervals': 'exploratory, no multiple-comparison significance claims',
    }
    return result


def robustness(out):
    source = read_jsonl(ROOT / 'data/korean_test100.jsonl')
    expected = {r['key']: r for r in source}
    R = {
        'status': 'post-hoc serialization robustness on previously inspected TEST50',
        'semantic_pairs': 50,
        'presentations_per_cell': 100,
        'cells': {},
        'paired_adapter_minus_base': {},
    }
    cells = {}
    for serial in ['plain', 'native']:
        for sys in ['base', 'adapter']:
            rows = load(out, 'robustness', sys, serial, expected)
            cells[(serial, sys)] = rows
            summary = {}
            for orient in ['original', 'swapped']:
                dd = [r for r in rows if r['orientation'] == orient]
                valid = [r for r in dd if r['valid']]
                summary[orient] = {
                    'n': len(dd),
                    'strict_accuracy': sum(r['correct'] for r in dd) / len(dd),
                    'invalid_count': len(dd) - len(valid),
                    'valid_only_accuracy': (
                        sum(r['correct'] for r in valid) / len(valid) if valid else None
                    ),
                    'valid_only_n': len(valid),
                    'label_logprob_accuracy': sum(r['logprob_correct'] for r in dd)
                    / len(dd),
                }
            a = {
                r['semantic_pair_id']: r for r in rows if r['orientation'] == 'original'
            }
            b = {
                r['semantic_pair_id']: r for r in rows if r['orientation'] == 'swapped'
            }
            summary['semantic_swap_consistency'] = (
                sum(
                    a[k]['valid']
                    and b[k]['valid']
                    and a[k]['prediction'] != b[k]['prediction']
                    for k in a
                )
                / 50
            )
            summary['generated_at_limit'] = sum(r['reached_token_limit'] for r in rows)
            R['cells'][serial + '_' + sys] = summary

        if any(
            a['serialized_ids_sha256'] != b['serialized_ids_sha256']
            for a, b in zip(cells[(serial, 'base')], cells[(serial, 'adapter')])
        ):
            raise ValueError('Base/adapter tokenization differs')
        a = {
            r['semantic_pair_id']: r
            for r in cells[(serial, 'base')]
            if r['orientation'] == 'original'
        }
        b = {
            r['semantic_pair_id']: r
            for r in cells[(serial, 'adapter')]
            if r['orientation'] == 'original'
        }
        ids = sorted(a)
        d = {
            k: np.array([int(b[i][k]) - int(a[i][k]) for i in ids])
            for k in ['correct', 'logprob_correct']
        }
        rng = np.random.default_rng(2026)
        ix = rng.integers(0, 50, size=(10000, 50))
        R['paired_adapter_minus_base'][serial] = {
            k: {
                'observed': float(v.mean()),
                'paired_semantic_pair_bootstrap_95': np.percentile(
                    v[ix].mean(axis=1), [2.5, 97.5]
                ).tolist(),
                'n': 50,
                'seed': 2026,
                'replicates': 10000,
            }
            for k, v in d.items()
        }
    return R


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['robustness', 'transfer'])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    validate_package()
    out = Path(args.out).expanduser().resolve()
    R = robustness(out) if args.task == 'robustness' else transfer(out)
    (out / f'{args.task}_analysis.json').write_text(
        json.dumps(R, indent=2, ensure_ascii=False, allow_nan=False)
    )
    print(json.dumps(R, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
