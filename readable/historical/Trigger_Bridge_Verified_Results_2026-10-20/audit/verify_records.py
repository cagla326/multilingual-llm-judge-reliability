#!/usr/bin/env python3
"""Independent reconstruction from raw bridge outputs; no GPU inference."""
from pathlib import Path
import collections, csv, hashlib, json, math, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'frozen_run_package'
RUN = ROOT / 'results_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def digest(x):
    return hashlib.sha256(
        json.dumps(x, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def near(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        [near(a[k], b[k]) for k in a]
    elif isinstance(a, list):
        assert len(a) == len(b)
        [near(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def score(ls):
    if not ls:
        return 0.0
    n = 100 * sum(v == 'entailment' for v in ls) / len(ls)
    return min(n, 25) if 'contradiction' in ls else min(n, 50) if 'neutral' in ls else n


def combine(b, c):
    if b >= 75 and c <= 50:
        return (b + c) / 2
    if b <= 50 and c >= 75:
        return (b + c) / 2
    return b


def stats(y, s):
    y = np.array(y)
    s = np.array(s)
    e = s - y
    return dict(
        n=len(y),
        spearman=float(spearmanr(y, s).statistic),
        kendall=float(kendalltau(y, s).statistic),
        pearson=float(pearsonr(y, s).statistic),
        mae=float(abs(e).mean()),
        rmse=float(np.sqrt((e**2).mean())),
        signed_bias=float(e.mean()),
        extreme_over_rate=float((e >= 50).mean()),
        extreme_under_rate=float((e <= -50).mean()),
    )


def main():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    freeze = json.loads((RUN / 'RUN_FREEZE.json').read_text())
    published = json.loads((RUN / 'FINAL_ANALYSIS.json').read_text())
    assert (
        sha(ROOT / 'PROTOCOL.json')
        == sha(PKG / 'PROTOCOL.json')
        == freeze['protocol_sha256']
    )
    for n, h in protocol['package_hashes'].items():
        assert sha(PKG / n) == h, n
    assert sha(RUN / 'ENCODED.json') == freeze['encoded_sha256']
    enc = json.loads((RUN / 'ENCODED.json').read_text())
    meta = read(PKG / 'data/examples.jsonl')
    claims = read(PKG / 'data/claims.jsonl')
    byclaim = {r['key']: r for r in claims}
    byid = {r['row_id']: r for r in meta}
    assert (
        len(meta) == len(byid) == 900
        and len(claims) == len(byclaim) == len(enc) == 2489
    )
    assert (
        collections.Counter(r['lang'] for r in meta)
        == collections.Counter({l: 20 for l in {r['lang'] for r in meta}})
        and len({r['lang'] for r in meta}) == 45
    )
    with (PKG / 'data/original_final900.csv').open() as f:
        original = {r['row_id']: r for r in csv.DictReader(f)}
    classes = ['entailment', 'contradiction', 'neutral']
    pred_counts = {'base': collections.Counter(), 'trained': collections.Counter()}
    labeltrans = collections.Counter()
    runtimes = collections.Counter()
    allkeys = []
    reconstructed = []
    deltas = []
    tokenlengths = []
    changed = []
    files = list((RUN / 'rows').glob('*.json'))
    assert len(files) == 900
    for ex in meta:
        orig = original[ex['row_id']]
        cs = json.loads(orig['claims'])
        assert ex['claim_keys'] == [f"{ex['row_id']}:{i}" for i in range(len(cs))]
        assert (
            ex['human'] == float(orig['human_0_100'])
            and ex['baseline'] == float(orig['gemma_fullsource_score'])
            and ex['original_final'] == float(orig['final_symmetric_extreme_average'])
        )
        assert ex['lang'] == orig['lang'] and ex['system'] == orig['system']
        oldlabels = json.loads(orig['claim_labels'])
        assert len(oldlabels) == len(cs)
        oldval = {
            'fully_supported': 100,
            'partially_supported': 50,
            'unsupported': 0,
            'contradicted': 0,
        }
        p = sum(oldval[x] for x in oldlabels) / len(oldlabels) if oldlabels else 0
        cap = (
            min(p, 25)
            if 'contradicted' in oldlabels
            else (
                min(p, 50)
                if 'unsupported' in oldlabels
                else min(p, 75) if 'partially_supported' in oldlabels else p
            )
        )
        near(cap, float(orig['claim_capped_score']))
        near(combine(ex['baseline'], cap), ex['original_final'])
        path = RUN / 'rows' / f'row_{int(ex["row_id"]):06d}.json'
        r = json.loads(path.read_text())
        assert (
            r['row_id'] == ex['row_id']
            and r['protocol_sha256'] == freeze['protocol_sha256']
        )
        assert published['input_record_hashes'][str(path.relative_to(RUN))] == sha(path)
        assert [v['key'] for v in r['claims']] == ex['claim_keys']
        rt = r['runtime']
        assert (
            rt['revision'] == freeze['revision']
            and rt['versions'] == freeze['versions']
            and rt['same_hardware_within_each_base_trained_pair']
            and rt['logits_to_keep'] == 1
            and not rt['use_cache']
        )
        runtimes[rt['gpu']] += 1
        labels = {'base': [], 'trained': []}
        for i, v in enumerate(r['claims']):
            cr = byclaim[v['key']]
            assert (
                cr['row_id'] == ex['row_id']
                and cr['premise'] == orig['translated_source_en']
                and cr['hypothesis'] == cs[i]
            )
            e = enc[v['key']]
            assert e['prompt_sha256'] == v['prompt_sha256'] == digest(
                cr['prompt']
            ) and e['input_ids_sha256'] == v['input_ids_sha256'] == digest(e['ids'])
            allkeys.append(v['key'])
            tokenlengths.append(len(e['ids']))
            for system in labels:
                lp = v[system]['restricted_logprobs']
                assert (
                    set(lp) == set(classes)
                    and all(math.isfinite(lp[c]) for c in classes)
                    and abs(sum(math.exp(lp[c]) for c in classes) - 1) < 1e-5
                )
                label = max(classes, key=lambda c: lp[c])
                assert label == v[system]['label']
                labels[system].append(label)
                pred_counts[system][label] += 1
            labeltrans[(labels['base'][-1], labels['trained'][-1])] += 1
        b = score(labels['base'])
        n = score(labels['trained'])
        row = dict(
            row_id=ex['row_id'],
            lang=ex['lang'],
            human=ex['human'],
            holistic=ex['baseline'],
            original4_selective=ex['original_final'],
            base3_capped=b,
            trained3_capped=n,
            base3_selective=combine(ex['baseline'], b),
            trained3_selective=combine(ex['baseline'], n),
            num_claims=len(cs),
        )
        reconstructed.append(row)
        if not math.isclose(
            row['trained3_selective'], row['original4_selective'], abs_tol=1e-9
        ):
            changed.append(
                {
                    **row,
                    'error_change': abs(row['trained3_selective'] - row['human'])
                    - abs(row['original4_selective'] - row['human']),
                    'source': orig['input'],
                    'summary': orig['summary'],
                    'translated_source_en': orig['translated_source_en'],
                    'claims': cs,
                    'base3_labels': labels['base'],
                    'trained3_labels': labels['trained'],
                }
            )
    assert (
        len(allkeys) == len(set(allkeys)) == 2489
        and max(tokenlengths) == freeze['max_prompt_tokens'] == 4217
        and sum(x > 221 for x in tokenlengths)
        == freeze['prompts_above_training_max221']
        == 2482
    )
    near(reconstructed, read(RUN / 'FINAL_ROWS.jsonl'))
    cols = [
        'holistic',
        'original4_selective',
        'base3_capped',
        'trained3_capped',
        'base3_selective',
        'trained3_selective',
    ]
    y = np.array([r['human'] for r in reconstructed])
    scores = {k: np.array([r[k] for r in reconstructed]) for k in cols}
    metrics = {k: stats(y, v) for k, v in scores.items()}
    near(metrics, published['metrics'])
    near(
        metrics['trained3_selective']['spearman']
        - metrics['original4_selective']['spearman'],
        published['primary_spearman_delta'],
    )
    for ref in ['original4_selective', 'base3_selective']:
        en = abs(scores['trained3_selective'] - y)
        eo = abs(scores[ref] - y)
        counts = dict(
            unchanged_scores=int(
                np.isclose(
                    scores['trained3_selective'], scores[ref], atol=1e-9, rtol=0
                ).sum()
            ),
            lower_absolute_error=int((en < eo - 1e-9).sum()),
            higher_absolute_error=int((en > eo + 1e-9).sum()),
            equal_absolute_error=int((abs(en - eo) <= 1e-9).sum()),
        )
        near(counts, published['changes_vs_references'][ref])
    # Different implementation: materialize sampled row indices, compute reference rank correlations.
    langs = sorted({r['lang'] for r in meta})
    idx = [np.array([i for i, r in enumerate(meta) if r['lang'] == l]) for l in langs]
    rng = np.random.default_rng(20261019)
    draws = {r: [] for r in ['original4_selective', 'base3_selective']}
    for _ in range(2000):
        ix = np.concatenate([idx[i] for i in rng.integers(0, 45, size=45)])
        yy = y[ix]
        nn = scores['trained3_selective'][ix]
        nr = float(spearmanr(yy, nn).statistic)
        ne = nn - yy
        for ref in draws:
            ss = scores[ref][ix]
            se = ss - yy
            draws[ref].append(
                [
                    nr - float(spearmanr(yy, ss).statistic),
                    float(abs(ne).mean() - abs(se).mean()),
                    float(ne.mean() - se.mean()),
                    float((ne <= -50).mean() - (se <= -50).mean()),
                    float((ne >= 50).mean() - (se >= 50).mean()),
                ]
            )
    cis = {}
    supp = {}
    for ref, dd in draws.items():
        ci = np.quantile(dd, [0.025, 0.975], axis=0)
        key = 'trained3_selective minus ' + ref
        pci = published['paired_language_cluster_intervals'][key]
        for i, n in enumerate(
            [
                'spearman_delta_95_percentile_ci',
                'mae_delta_95_percentile_ci',
                'bias_delta_95_percentile_ci',
            ]
        ):
            near(ci[:, i].tolist(), pci[n])
        cis[key] = pci
        supp[key] = {
            'extreme_under_rate_delta_95_percentile_ci': ci[:, 3].tolist(),
            'extreme_over_rate_delta_95_percentile_ci': ci[:, 4].tolist(),
            'status': 'supplementary post-result audit, not the frozen primary endpoint',
        }
    for task, job in [('smoke', '469204'), ('infer', '469206')]:
        assert 'COMPLETE ' + task in (ROOT / f'trigger_bridge_{job}.out').read_text()
        assert 'Traceback' not in (ROOT / f'trigger_bridge_{job}.err').read_text()
    assert sha(RUN / 'RUN_FREEZE.json') == published['freeze_sha256']
    audit = {
        'status': 'VERIFIED_FROM_RAW_RECORDS_NO_GPU_RERUN',
        'package_hashes_checked': len(protocol['package_hashes']),
        'rows': 900,
        'claims': 2489,
        'model_decisions': 4978,
        'runtime_rows_by_gpu': dict(runtimes),
        'primary_spearman_delta': published['primary_spearman_delta'],
        'metrics': metrics,
        'reproduced_intervals': cis,
        'supplementary_error_rate_intervals': supp,
        'claim_label_counts': {k: dict(v) for k, v in pred_counts.items()},
        'claim_label_transitions': {
            a + ' -> ' + b: n for (a, b), n in labeltrans.items()
        },
        'changes_vs_references': published['changes_vs_references'],
        'empty_claim_rows': [r['row_id'] for r in meta if not r['claim_keys']],
        'limits': [
            'Primary ranking superiority over original4 is not established: interval includes zero.',
            'Secondary MAE gain over original4 is supported under language resampling; extreme under-scoring increases.',
            'Matched trained3/base3 ranking effect is positive; its MAE interval includes zero and bias increases.',
            'Same inspected benchmark, frozen after earlier development; not external confirmation.',
            'English longer-context inference from Korean short-NLI training; no isolated mechanism claim.',
            'Weights are not in this upload; loader checks and separately author-reported Mac hashes link checkpoint identity, not local independent byte inspection.',
            'GPU logits and actual tokenization not regenerated.',
            'Original historical verifier immutable revision remains incompletely recovered.',
        ],
    }
    (ROOT / 'audit/VERIFICATION.json').write_text(json.dumps(audit, indent=2))
    (ROOT / 'audit/CHANGED_CASES.json').write_text(
        json.dumps(changed, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
