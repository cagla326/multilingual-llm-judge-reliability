"""Run from workspace root; check recorded predictions against frozen inputs and raw outputs."""

from pathlib import Path
import sys, json, hashlib, collections, re, math, importlib.util
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr, binomtest

BASE = Path.cwd()
PKG = BASE / 'Korean_Extension_2026-10-19/package'
OUT = BASE / 'extension_results_2026-10-19/results_v1'
DEST = OUT.parent / 'audit'
sys.path.insert(0, str(PKG))
import run_extension as runner
import analyze_extension as analyzer

read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
freeze = read(OUT / 'RUN_FREEZE.json')
protocol = read(OUT.parent / 'PROTOCOL.json')
assert sha(OUT.parent / 'PROTOCOL.json') == freeze['protocol_sha256']
assert protocol == read(PKG / 'PROTOCOL.json')
for n, h in freeze['package_hashes'].items():
    assert sha(PKG / n) == h, n
checks = {
    'frozen_package_hashes_match': len(freeze['package_hashes']),
    'uploaded_protocol_matches': True,
    'cells': {},
    'completion_logs': {},
}
for task, job in [('robustness', 468973), ('transfer', 468974)]:
    log = (OUT.parent / f'ko_extension_{job}.out').read_text()
    assert f'COMPLETE {task}' in log
    checks['completion_logs'][task] = True
    meta = read(OUT / f'{task}_runtime.json')
    assert meta['revision'] == freeze['revision']
    assert meta['dtype'] == 'bfloat16'
    assert meta['python'] == freeze['python']
    assert meta['gpu'] == 'NVIDIA H200 NVL'
    source = rows(
        PKG
        / (
            'data/korean_test100.jsonl'
            if task == 'robustness'
            else 'data/nonkorean880.jsonl'
        )
    )
    expected = {r['key']: r for r in source}
    assert len(expected) == len(source)
    serials = ['plain', 'native'] if task == 'robustness' else ['native']
    for serial in serials:
        data = {}
        for system in ['base', 'adapter']:
            rr = analyzer.load(OUT, task, system, serial, expected)
            data[system] = rr
            for r in rr:
                assert r['generated_tokens'] <= r['output_limit']
                assert r['reached_token_limit'] == (
                    r['generated_tokens'] == r['output_limit']
                )
                assert r['output_limit'] == (4 if task == 'robustness' else 256)
                if task == 'transfer':
                    p = runner.parse_score(r['raw_response'])
                    for k, v in p.items():
                        assert r[k] == v, (r['key'], k)
                else:
                    pred = r['raw_response'].strip()
                    pred = pred if pred in ['A', 'B'] else None
                    assert r['prediction'] == pred
                    assert r['valid'] == (pred is not None)
                    assert r['correct'] == (pred == r['gold'])
                    a, b = r['logprob_A'], r['logprob_B']
                    assert math.isfinite(a) and math.isfinite(b)
                    lp = 'A' if a > b else 'B' if b > a else None
                    assert r['logprob_prediction'] == lp
                    assert r['logprob_correct'] == (lp == r['gold'])
            checks['cells'][f'{task}_{system}_{serial}'] = {
                'rows': len(rr),
                'raw_fields_verified': True,
                'max_prompt_tokens': max(r['prompt_tokens'] for r in rr),
                'max_generated_tokens': max(r['generated_tokens'] for r in rr),
            }
        assert all(
            a['serialized_ids_sha256'] == b['serialized_ids_sha256']
            for a, b in zip(data['base'], data['adapter'])
        )


# Regenerate both analyses with the frozen implementation and compare numeric outputs.
def compare(a, b, path=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i))
    elif isinstance(a, float):
        assert np.isclose(a, b, atol=1e-11, rtol=1e-11), (path, a, b)
    else:
        assert a == b, (path, a, b)


for task, fn in [('robustness', analyzer.robustness), ('transfer', analyzer.transfer)]:
    recomputed = fn(OUT)
    compare(recomputed, read(OUT / f'{task}_analysis.json'))
    checks[task + '_analysis_reproduced'] = True
# Independent transfer metrics and descriptive change counts.
b = rows(OUT / 'transfer_base_native.jsonl')
a = rows(OUT / 'transfer_adapter_native.jsonl')
assert len(b) == len(a) == 880
Y = np.array([r['human'] for r in b])
B = np.array([r['score'] for r in b])
A = np.array([r['score'] for r in a])
L = np.array([r['lang'] for r in b])
assert len(set(L)) == 44 and 'korean' not in set(L)
assert set(collections.Counter(L).values()) == {20}
independent = {}
for name, S in [('base', B), ('adapter', A)]:
    e = S - Y
    independent[name] = {
        'n': len(Y),
        'spearman': float(spearmanr(Y, S).statistic),
        'kendall': float(kendalltau(Y, S).statistic),
        'pearson': float(pearsonr(Y, S).statistic),
        'mae': float(abs(e).mean()),
        'rmse': float(np.sqrt((e * e).mean())),
        'bias': float(e.mean()),
        'extreme_over': float((e >= 50).mean()),
        'extreme_under': float((e <= -50).mean()),
    }
    for k, v in independent[name].items():
        assert np.isclose(
            v, read(OUT / 'transfer_analysis.json')['paired_metrics'][name][k]
        )
checks['transfer_independent_metrics'] = independent
checks['transfer_changes'] = {
    'unchanged_scores': int((A == B).sum()),
    'changed_scores': int((A != B).sum()),
    'closer_to_human': int((abs(A - Y) < abs(B - Y)).sum()),
    'farther_from_human': int((abs(A - Y) > abs(B - Y)).sum()),
    'same_absolute_error': int((abs(A - Y) == abs(B - Y)).sum()),
}
# Paired Korean contrasts: present both orientations together, without treating 100 presentations as independent pairs.
supp = {}
for serial in ['plain', 'native']:
    bb = rows(OUT / f'robustness_base_{serial}.jsonl')
    aa = rows(OUT / f'robustness_adapter_{serial}.jsonl')
    bm = {(r['semantic_pair_id'], r['orientation']): r for r in bb}
    am = {(r['semantic_pair_id'], r['orientation']): r for r in aa}
    ids = sorted({r['semantic_pair_id'] for r in bb})
    supp[serial] = {}
    for orient in ['original', 'swapped']:
        wins = sum(
            not bm[i, orient]['correct'] and am[i, orient]['correct'] for i in ids
        )
        losses = sum(
            bm[i, orient]['correct'] and not am[i, orient]['correct'] for i in ids
        )
        supp[serial][orient] = {
            'base_correct': sum(bm[i, orient]['correct'] for i in ids),
            'adapter_correct': sum(am[i, orient]['correct'] for i in ids),
            'adapter_only_correct': wins,
            'base_only_correct': losses,
            'exact_paired_binomial_p_descriptive': (
                float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else 1.0
            ),
        }
    d = np.array(
        [
            sum(
                int(am[i, o]['correct']) - int(bm[i, o]['correct'])
                for o in ['original', 'swapped']
            )
            / 2
            for i in ids
        ]
    )
    rng = np.random.default_rng(2026)
    ix = rng.integers(0, 50, size=(10000, 50))
    supp[serial]['both_orientations_pair_clustered_supplementary'] = {
        'delta': float(d.mean()),
        'percentile95': np.percentile(d[ix].mean(axis=1), [2.5, 97.5]).tolist(),
        'semantic_pairs': 50,
        'replicates': 10000,
        'seed': 2026,
        'interpretation': 'post-hoc supplementary; preserve uploaded original-orientation interval and historical primary results',
    }
checks['korean_supplementary'] = supp
checks['limitations'] = [
    'Raw log probabilities and token-ID hashes are recorded, not recomputed without GPU/model weights here.',
    'Source/prompt/input alignment verified against frozen packaged inputs; cross-language semantic leakage cannot be ruled out.',
    'Frozen Korean analyzer computes bootstrap for original orientation only; supplementary audit additionally keeps both orientations within each resampled semantic pair.',
    'Small post-hoc bootstrap intervals are not confirmatory significance evidence; original native comparison has 4 discordant wins and 0 losses, exact paired p=0.125.',
    'Extension does not establish immutable historical revision equivalence or external validation.',
]
(DEST / 'VERIFICATION.json').write_text(
    json.dumps(checks, indent=2, ensure_ascii=False, allow_nan=False)
)
print(
    json.dumps(
        {k: v for k, v in checks.items() if k not in ['cells', 'limitations']}, indent=2
    )
)
