#!/usr/bin/env python3
"""Complete paired audit. Official three-way KLUE accuracy is not computed."""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest
from run_klue import ROOT, validate_package, read_jsonl, score_record


def load(out, task, system):
    expected = read_jsonl(
        ROOT / ('data/ablation.jsonl' if task == 'ablation' else 'data/evaluate.jsonl')
    )
    ee = {r['key']: r for r in expected}
    rr = read_jsonl(out / (task + '_' + system + '.jsonl'))
    actual = {r['key']: r for r in rr}
    if len(actual) != len(rr) or set(actual) != set(ee):
        raise ValueError(
            f'Incomplete/duplicate {task}/{system}: {len(actual)} of {len(ee)}'
        )
    ordered = []
    for src in expected:
        r = actual[src['key']]
        if r['system'] != system:
            raise ValueError('Wrong system')
        if r['prompt_sha256'] != hashlib.sha256(src['prompt'].encode()).hexdigest():
            raise ValueError('Prompt hash mismatch')
        for k, v in src.items():
            if k != 'prompt' and r.get(k) != v:
                raise ValueError('Changed source metadata ' + k)
        checked = score_record(
            r['raw_response'], r['gold'], r['logprob_A'], r['logprob_B']
        )
        for k, v in checked.items():
            if r[k] != v:
                raise ValueError(
                    'Stored result disagrees with raw/probability record: ' + k
                )
        ordered.append(r)
    return ordered


def bootstrap(values, seed=2026):
    values = np.array(values, dtype=float)
    support, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    # Count bootstrap exactly resamples the empirical paired-difference distribution.
    draws = rng.multinomial(len(values), counts / len(values), size=10000).dot(
        support
    ) / len(values)
    return {
        'observed': float(values.mean()),
        'percentile95': np.percentile(draws, [2.5, 97.5]).tolist(),
        'replicates': 10000,
        'seed': seed,
        'unit': 'semantic premise; empirical paired-difference count bootstrap',
    }


def task_summary(out, task):
    data = {s: load(out, task, s) for s in ['base', 'adapter']}
    a = data['base']
    b = data['adapter']
    if any(
        x['serialized_ids_sha256'] != y['serialized_ids_sha256'] for x, y in zip(a, b)
    ):
        raise ValueError('Model inputs differ')
    result = {
        'context': 'provided' if task == 'evaluate' else 'omitted',
        'presentations_per_system': len(a),
        'semantic_pairs': len(a) // 2,
        'systems': {},
        'paired_contrasts': {},
        'by_source': [],
    }
    maps = {
        s: {(r['semantic_pair_id'], r['orientation']): r for r in data[s]} for s in data
    }
    ids = sorted({r['semantic_pair_id'] for r in a})
    for s in data:
        result['systems'][s] = {}
        for ori in ['original', 'swapped']:
            rows = [maps[s][i, ori] for i in ids]
            valid = sum(r['valid'] for r in rows)
            result['systems'][s][ori] = {
                'n': len(rows),
                'correct': sum(r['correct'] for r in rows),
                'accuracy': sum(r['correct'] for r in rows) / len(rows),
                'valid': valid,
                'invalid': len(rows) - valid,
                'valid_only_accuracy': (
                    sum(r['correct'] for r in rows) / valid if valid else None
                ),
                'label_logprob_accuracy': sum(r['logprob_correct'] for r in rows)
                / len(rows),
                'at_output_limit': sum(r['reached_token_limit'] for r in rows),
            }
        result['systems'][s]['semantic_swap_consistency'] = sum(
            maps[s][i, 'original']['valid']
            and maps[s][i, 'swapped']['valid']
            and maps[s][i, 'original']['prediction']
            != maps[s][i, 'swapped']['prediction']
            for i in ids
        ) / len(ids)
    for ori in ['original', 'swapped']:
        wins = sum(
            not maps['base'][i, ori]['correct'] and maps['adapter'][i, ori]['correct']
            for i in ids
        )
        losses = sum(
            maps['base'][i, ori]['correct'] and not maps['adapter'][i, ori]['correct']
            for i in ids
        )
        diff = [
            int(maps['adapter'][i, ori]['correct'])
            - int(maps['base'][i, ori]['correct'])
            for i in ids
        ]
        result['paired_contrasts'][ori] = {
            'strict_difference': bootstrap(diff),
            'adapter_only_correct': wins,
            'base_only_correct': losses,
            'both_correct': sum(
                maps['base'][i, ori]['correct'] and maps['adapter'][i, ori]['correct']
                for i in ids
            ),
            'both_incorrect': sum(
                not maps['base'][i, ori]['correct']
                and not maps['adapter'][i, ori]['correct']
                for i in ids
            ),
            'exact_two_sided_paired_binomial_p': (
                float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else 1.0
            ),
            'label_logprob_difference': bootstrap(
                [
                    int(maps['adapter'][i, ori]['logprob_correct'])
                    - int(maps['base'][i, ori]['logprob_correct'])
                    for i in ids
                ]
            ),
        }
    result['paired_contrasts']['both_orientations_supplementary'] = bootstrap(
        [
            sum(
                int(maps['adapter'][i, o]['correct'])
                - int(maps['base'][i, o]['correct'])
                for o in ['original', 'swapped']
            )
            / 2
            for i in ids
        ]
    )
    for src in sorted({r['source'] for r in a}):
        ss = [i for i in ids if maps['base'][i, 'original']['source'] == src]
        result['by_source'].append(
            {
                'source': src,
                'n': len(ss),
                **{
                    s: {
                        ori: sum(maps[s][i, ori]['correct'] for i in ss) / len(ss)
                        for ori in ['original', 'swapped']
                    }
                    for s in data
                },
            }
        )
    return result, maps, ids


def analyze(out):
    validate_package()
    spec = json.loads((ROOT / 'PROTOCOL.json').read_text())
    f = json.loads((out / 'RUN_FREEZE.json').read_text())
    if (
        f['protocol_sha256']
        != hashlib.sha256((ROOT / 'PROTOCOL.json').read_bytes()).hexdigest()
    ):
        raise ValueError('Freeze/protocol mismatch')
    full, fm, ids = task_summary(out, 'evaluate')
    abl, am, ids2 = task_summary(out, 'ablation')
    assert ids == ids2
    rt = [
        json.loads((out / (task + '_runtime.json')).read_text())
        for task in ['evaluate', 'ablation']
    ]
    for k in [
        'gpu',
        'cuda',
        'dtype',
        'revision',
        'python',
        'max_new_tokens',
        'do_sample',
        'prompt_serialization',
    ]:
        if rt[0][k] != rt[1][k]:
            raise ValueError(
                'Hardware/runtime differs between context conditions: ' + k
            )
    interaction = [
        (
            int(fm['adapter'][i, 'original']['correct'])
            - int(fm['base'][i, 'original']['correct'])
        )
        - (
            int(am['adapter'][i, 'original']['correct'])
            - int(am['base'][i, 'original']['correct'])
        )
        for i in ids
    ]
    return {
        'status': 'COMPLETE_PAIRED_EXTERNAL_TASK_EVALUATION',
        'dataset': 'KLUE-NLI published development data; human-authored hypotheses and published majority labels',
        'official_KLUE_three_way_score': False,
        'primary': 'adapter minus base strict original-orientation accuracy with premise provided; invalid answers count as incorrect',
        'primary_result': full['paired_contrasts']['original'],
        'full_context': full,
        'premise_omitted_secondary_control': abl,
        'secondary_context_gain_interaction': bootstrap(interaction),
        'interpretation_limits': [
            'One prespecified primary exact paired test. Other endpoints and subgroup analyses are exploratory.',
            'A/B pairwise reformulation excludes neutral and is not official three-way KLUE-NLI evaluation.',
            'Public development split external to adapter construction, not hidden test; pretraining exposure is unknown.',
            'Premise-omitted performance measures candidate cues/prior knowledge; do not label it source-grounded accuracy.',
            'Semantic-pair resampling does not account for unknown clustering of multiple premises within an original document.',
            'Source-ablation interaction alone does not prove a particular internal reasoning mechanism.',
            'No retraining, prompt selection or result-based subset selection.',
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    result = analyze(out)
    (out / 'KLUE_ANALYSIS.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    )
    print(
        json.dumps(
            {
                'status': result['status'],
                'primary': result['primary_result'],
                'systems': result['full_context']['systems'],
                'source_control_systems': result['premise_omitted_secondary_control'][
                    'systems'
                ],
                'secondary_context_gain_interaction': result[
                    'secondary_context_gain_interaction'
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == '__main__':
    main()
