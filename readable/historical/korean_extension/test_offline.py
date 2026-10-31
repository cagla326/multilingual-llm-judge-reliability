"""Offline integrity and analysis tests. Contains no model inference."""

import json, tempfile
from pathlib import Path
import numpy as np
import run_extension as run
import analyze_extension as ana


def main():
    spec = run.validate_package()
    assert spec['n_transfer'] == 880
    # Parsing distinctions matter: off-anchor scores stay, malformed missing-score recovery is labelled,
    # empty output is unavailable and out-of-range numbers are not silently clipped.
    assert run.parse_score('{"score":90,"reason":"ok"}')['score'] == 90
    assert run.parse_score('{"score":90}')['parser_anchor'] is False
    assert (
        run.parse_score('```json\n{"score":75,"reason":"a "bad" quote"}\n```')[
            'recovered'
        ]
        is True
    )
    assert run.parse_score('no score')['score'] is None
    assert run.parse_score('{"score":125}')['score'] is None
    assert run.parse_score('{"score":0}')['score'] == 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sources = run.read_jsonl(run.ROOT / 'data/nonkorean880.jsonl')
        for sys in ['base', 'adapter']:
            rows = []
            for r in sources:
                rec = {k: v for k, v in r.items() if k != 'prompt'}
                rec.update(
                    system=sys,
                    serialization='native',
                    score=(100 - r['human']) if sys == 'base' else r['human'],
                    parser_anchor=False,
                    recovered=False,
                    reached_token_limit=False,
                    prompt_sha256=__import__('hashlib')
                    .sha256(r['prompt'].encode())
                    .hexdigest(),
                    serialized_ids_sha256='same-synthetic-tokenization',
                )
                rows.append(rec)
            (out / f'transfer_{sys}_native.jsonl').write_text(
                ''.join(json.dumps(r) + '\n' for r in rows)
            )
        report = ana.transfer(out)
        assert report['paired_n'] == 880 and report['paired_languages'] == 44
        assert np.isclose(report['observed_delta_adapter_minus_base']['spearman'], 2)
        assert all(
            np.isclose(x, 2)
            for x in report['paired_language_cluster_bootstrap'][
                'percentile_95_intervals'
            ]['spearman']
        )
        # A partial run must fail, rather than present a favourable subset as complete.
        path = out / 'transfer_adapter_native.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1]) + '\n')
        try:
            ana.transfer(out)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete run was accepted')
        sources = run.read_jsonl(run.ROOT / 'data/korean_test100.jsonl')
        for serial in ['plain', 'native']:
            for sys in ['base', 'adapter']:
                rows = []
                for r in sources:
                    rec = {k: v for k, v in r.items() if k != 'prompt'}
                    pred = r['gold'] if sys == 'adapter' else None
                    rec.update(
                        system=sys,
                        serialization=serial,
                        prediction=pred,
                        valid=pred is not None,
                        correct=pred == r['gold'],
                        logprob_correct=sys == 'adapter',
                        reached_token_limit=False,
                        prompt_sha256=__import__('hashlib')
                        .sha256(r['prompt'].encode())
                        .hexdigest(),
                        serialized_ids_sha256='same-synthetic-tokenization',
                    )
                    rows.append(rec)
                (out / f'robustness_{sys}_{serial}.jsonl').write_text(
                    ''.join(json.dumps(r) + '\n' for r in rows)
                )
        report = ana.robustness(out)
        assert report['cells']['native_base']['original']['strict_accuracy'] == 0
        assert report['cells']['native_adapter']['semantic_swap_consistency'] == 1
        assert report['paired_adapter_minus_base']['native']['correct']['observed'] == 1
    print(
        'PASS: package hashes, parser distinctions, full-cohort analyses, paired uncertainty, incomplete-run refusal, swap consistency. No GPU inference executed.'
    )


if __name__ == '__main__':
    main()
