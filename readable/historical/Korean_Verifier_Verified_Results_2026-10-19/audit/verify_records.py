#!/usr/bin/env python3
"""Artifact audit, no model inference. Run with NumPy/SciPy from the archive root."""
from pathlib import Path
import collections, hashlib, json, math, sys
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'frozen_run_package'
RUN = ROOT / 'run_v1'
sys.path.insert(0, str(PKG))
from data_checks import validate_data, read_rows
from verifier import metrics
from analyze import checked, clustered_ci

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        [near(a[k], b[k]) for k in a]
    elif isinstance(a, list):
        assert len(a) == len(b)
        [near(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def class_stats(cm):
    cm = np.array(cm)
    out = {}
    for i, c in enumerate(['entailment', 'contradiction', 'neutral']):
        precision = cm[i, i] / cm[:, i].sum()
        recall = cm[i, i] / cm[i, :].sum()
        out[c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(2 * precision * recall / (precision + recall)),
            'correct': int(cm[i, i]),
            'support': int(cm[i, :].sum()),
            'predicted_count': int(cm[:, i].sum()),
        }
    return out


def main():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    freeze = json.loads((RUN / 'RUN_FREEZE.json').read_text())
    complete = json.loads((RUN / 'TRAIN_COMPLETE.json').read_text())
    encoded = json.loads((RUN / 'ENCODED.json').read_text())
    assert (
        sha(ROOT / 'PROTOCOL.json')
        == sha(PKG / 'PROTOCOL.json')
        == freeze['protocol_sha256']
    )
    for name, h in protocol['package_hashes'].items():
        assert sha(PKG / name) == h, name
    checks = validate_data()
    assert checks == freeze['data_check']
    assert sha(RUN / 'ENCODED.json') == freeze['encoded_sha256']
    lengths = []
    for split, es in encoded.items():
        refs = read_rows(PKG / 'data' / f'{split}.jsonl')
        assert len(refs) == len(es)
        for r, e in zip(refs, es):
            assert r['key'] == e['key']
            assert (
                len(e['answer_ids']) == 2
                and e['answer_ids'][0] == freeze['label_ids'][r['answer']]
            )
            assert e['prompt_ids'] and all(
                type(i) == int and i >= 0 for i in e['prompt_ids']
            )
            lengths.append(len(e['prompt_ids']) + len(e['answer_ids']))
    assert max(lengths) == freeze['max_full_training_tokens'] == 221
    # Independently recompute every row metric and validate every frozen key/hash.
    allmetrics = {}
    for split, names in [
        ('validation', ['dev_epoch_0', 'dev_epoch_1', 'dev_epoch_2']),
        ('internal_test', ['heldout_base', 'heldout_new']),
    ]:
        refs = read_rows(PKG / 'data' / f'{split}.jsonl')
        enc = {e['key']: e for e in encoded[split]}
        for name in names:
            system = 'base' if name == 'heldout_base' else 'new'
            rr = checked(read_rows(RUN / f'{name}.jsonl'), refs, system)
            for r in rr:
                assert (
                    r['input_ids_sha256']
                    == hashlib.sha256(
                        json.dumps(enc[r['key']]['prompt_ids']).encode()
                    ).hexdigest()
                )
            allmetrics[name] = metrics(rr)
    for i, h in enumerate(complete['dev_history']):
        near({k: v for k, v in h.items() if k != 'epoch'}, allmetrics[f'dev_epoch_{i}'])
        assert h['epoch'] == i
    chosen = min(
        range(3),
        key=lambda i: allmetrics[f'dev_epoch_{i}']['mean_restricted_label_nll'],
    )
    assert chosen == complete['selected_epoch'] == 2
    assert (
        complete['optimizer_steps'] == 750
        and not complete['new_adapter_is_noop_epoch0']
    )
    config = RUN / complete['selected_adapter_relative_path'] / 'adapter_config.json'
    assert sha(config) == complete['selected_adapter_hashes']['adapter_config.json']
    ac = json.loads(config.read_text())
    assert (
        ac['r'] == 8
        and ac['lora_alpha'] == 16
        and ac['lora_dropout'] == 0.05
        and ac['bias'] == 'none'
    )
    assert ac['base_model_name_or_path'].endswith(freeze['revision'])
    base = read_rows(RUN / 'heldout_base.jsonl')
    new = read_rows(RUN / 'heldout_new.jsonl')
    ref = read_rows(PKG / 'data/internal_test.jsonl')
    base = checked(base, ref, 'base')
    new = checked(new, ref, 'new')
    bm = metrics(base)
    nm = metrics(new)
    ci = clustered_ci(base, new)
    result = json.loads((RUN / 'VERIFIED_ASSESSMENT.json').read_text())
    near(result['base'], bm)
    near(result['new'], nm)
    near(result['paired_source_bootstrap'], ci)
    for name, h in result['inputs_sha256'].items():
        assert sha(RUN / name) == h
    near(result['accuracy_delta'], nm['accuracy'] - bm['accuracy'])
    near(result['macro_f1_delta'], nm['macro_f1'] - bm['macro_f1'])
    summary = json.loads((RUN / 'HELDOUT_SUMMARY.json').read_text())
    near(summary['base'], bm)
    near(summary['new'], nm)
    transitions = collections.Counter()
    bygenre = collections.defaultdict(lambda: collections.Counter())
    changed = []
    for b, n, r in zip(base, new, ref):
        assert b['key'] == n['key'] == r['key']
        bc = b['prediction'] == r['human_label']
        nc = n['prediction'] == r['human_label']
        label = (
            'both_correct'
            if bc and nc
            else (
                'new_only_correct'
                if nc
                else 'base_only_correct' if bc else 'both_incorrect'
            )
        )
        transitions[label] += 1
        bygenre[r['genre']]['n'] += 1
        bygenre[r['genre']]['base_correct'] += bc
        bygenre[r['genre']]['new_correct'] += nc
        if bc != nc:
            changed.append(
                {
                    **{
                        k: r[k]
                        for k in [
                            'key',
                            'genre',
                            'premise',
                            'hypothesis',
                            'human_label',
                        ]
                    },
                    'base_prediction': b['prediction'],
                    'new_prediction': n['prediction'],
                    'transition': label,
                }
            )
    # Reproduce cluster intervals independently with a different implementation.
    rng = np.random.default_rng(20261019)
    groups = sorted({r['premise_hash'] for r in ref})
    idx = {
        g: np.array([i for i, r in enumerate(ref) if r['premise_hash'] == g])
        for g in groups
    }
    delta = []
    for _ in range(2000):
        sample = np.concatenate(
            [idx[groups[i]] for i in rng.integers(0, len(groups), size=len(groups))]
        )
        bs = np.array([base[i]['prediction'] == ref[i]['human_label'] for i in sample])
        ns = np.array([new[i]['prediction'] == ref[i]['human_label'] for i in sample])
        delta.append(float(ns.mean() - bs.mean()))
    near(
        np.quantile(delta, [0.025, 0.975]).tolist(),
        ci['accuracy_delta_95_percentile_ci'],
    )
    for task, job in [('smoke', '469169'), ('train', '469198'), ('assess', '469201')]:
        assert f'COMPLETE {task}' in (ROOT / f'ko_verifier_{job}.out').read_text()
        assert 'Traceback' not in (ROOT / f'ko_verifier_{job}.err').read_text()
    runtimes = {
        t: json.loads((RUN / f'{t}_runtime.json').read_text())
        for t in ['smoke', 'train', 'assess']
    }
    assert runtimes['train'] == runtimes['assess']
    assert runtimes['train']['revision'] == freeze['revision']
    audit = {
        'status': 'UPLOADED_RECORDS_VERIFIED_NO_MODEL_INFERENCE_RERUN',
        'package_hashes_checked': len(protocol['package_hashes']),
        'protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
        'data': checks,
        'selected_epoch': chosen,
        'selected_by': 'validation restricted-label NLL; epoch1 has higher validation accuracy but was not the selection criterion',
        'dev_epoch_metrics': allmetrics,
        'heldout_transitions': dict(transitions),
        'per_class_base': class_stats(bm['confusion_true_rows_predicted_columns']),
        'per_class_new': class_stats(nm['confusion_true_rows_predicted_columns']),
        'accuracy_delta': nm['accuracy'] - bm['accuracy'],
        'macro_f1_delta': nm['macro_f1'] - bm['macro_f1'],
        'source_cluster_bootstrap': ci,
        'supplementary_exact_paired_p': float(
            binomtest(
                transitions['new_only_correct'],
                transitions['new_only_correct'] + transitions['base_only_correct'],
                0.5,
            ).pvalue
        ),
        'descriptive_genres': {k: dict(v) for k, v in bygenre.items()},
        'runtimes': runtimes,
        'weight_verification': 'Selected weight hash is recorded and inference code checks it on cluster; safetensors not included in upload, so bytes are not independently verified here.',
        'token_verification': 'Frozen encoded IDs and output hashes checked; actual tokenizer and GPU logits not rerun.',
        'limits': [
            'Internal KLUE-train subset, 200 premises; not official test or external generalization.',
            'One training seed and one prompt/label serialization.',
            'Same-domain supervised training compared with untuned base; not a controlled comparison of SFT versus historical DPO.',
            'No inference about multilingual summary scoring or final-trigger improvement.',
            'Model pretraining contamination and semantic paraphrase overlap not excluded.',
            'Neutral recall decreases despite improved neutral precision and F1.',
        ],
    }
    (ROOT / 'audit/VERIFICATION.json').write_text(json.dumps(audit, indent=2))
    (ROOT / 'audit/DISCORDANT_CASES.json').write_text(
        json.dumps(changed, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
