"""Saved-output verification; no model inference, new fit, or hypothesis test.

First run supplementary_controls/analyze_controls.py into a NEW directory.
Then pass that output directory here. Optional --manuscript validates the
printed table values in the companion source project and checks Figure 2.
"""

from pathlib import Path
import argparse, hashlib, importlib.util, json, re, subprocess, sys
import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b):
    return bool(np.allclose(a, b, rtol=0, atol=1e-10))


def table(s, label):
    for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', s, re.S):
        if '\\label{' + label + '}' in m.group():
            return m.group()
    raise AssertionError('Table missing: ' + label)


def row_numbers(t, label):
    for line in t.splitlines():
        if line.strip().startswith(label + ' &'):
            return [
                float(x)
                for x in re.findall(
                    r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?', line.split('&', 1)[1]
                )
            ]
    raise AssertionError('Row missing: ' + label)


def printed(t, label, values, digits):
    assert close(
        row_numbers(t, label), [float(f'{v:.{d}f}') for v, d in zip(values, digits)]
    ), (label, row_numbers(t, label), values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reproduced', type=Path, required=True)
    ap.add_argument('--manuscript', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--evidence-root', type=Path)
    a = ap.parse_args()
    root = (a.evidence_root or Path(__file__).resolve().parent).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    assert not (
        a.out / 'VERIFICATION.json'
    ).exists(), 'Choose a new verification output directory.'
    manifest = json.loads((root / 'EVIDENCE_MANIFEST.json').read_text())
    assert all(sha(root / f['path']) == f['sha256'] for f in manifest['files'])
    r = json.loads((a.reproduced / 'RESULTS.json').read_text())
    expected = json.loads(
        (root / 'supplementary_controls/results_v1/RESULTS.json').read_text()
    )
    assert r == expected
    module = root / 'supplementary_controls/analyze_controls.py'
    spec = importlib.util.spec_from_file_location('controls', module)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    scores = pd.read_csv(a.reproduced / 'additional_scores.csv').set_index('row_id')
    labels = pd.read_csv(
        root / 'supplementary_controls/data/additional_labels.csv'
    ).set_index('holdout_row')
    repaired = {}
    for p in sorted(
        (root / 'supplementary_controls/data/additional_rows').glob('row_*.json')
    ):
        x = json.loads(p.read_text())
        rid = int(x['holdout_row'])
        assert int(x['id']) == int(labels.loc[rid, 'id'])
        repaired[rid] = float(
            C.selective(float(x['H']['score']), float(x['NLI_R']['trained3']['capped']))
        )
    ordered = scores.sort_index()
    y = ordered.y.to_numpy()
    repair = C.metrics(y, np.array([repaired[k] for k in ordered.index]))
    systems = dict(r['additional'])
    systems['SV_NLI_repair'] = repair
    original = json.loads(
        (root / 'archival/additional_run/run_v1/RESULTS.json').read_text()
    )
    for new, old in [
        ('H', 'H'),
        ('SV', 'S0'),
        ('SV_NLI', 'S*_T'),
        ('SV_NLI_repair', 'S*_R'),
        ('SV_NLI_base', 'base3_T'),
    ]:
        for key, v in systems[new].items():
            if key != 'n':
                assert close(
                    v,
                    original['systems'][old][
                        'kendall_tau_b' if key == 'kendall' else key
                    ],
                ), (new, key, v)
    pd.DataFrame(systems).T.to_csv(a.out / 'additional_metrics_reproduced.csv')
    # Recompute within-language summaries from row-level score matrices as a separate check.
    lang = pd.read_csv(a.reproduced / 'within_language.csv')
    within = {}
    for system, g in lang.groupby('system'):
        vals = g.spearman.dropna().to_numpy()
        summary = dict(
            defined=len(vals),
            undefined=int(g.spearman.isna().sum()),
            median=float(np.median(vals)),
            q1=float(np.percentile(vals, 25)),
            q3=float(np.percentile(vals, 75)),
            positive=int((vals > 0).sum()),
        )
        within[system] = summary
        assert all(
            close(v, r['within_language'][system][k]) for k, v in summary.items()
        )
    report = {
        'status': 'SAVED_OUTPUT_REPRODUCTION_VERIFIED',
        'manifest_payload_files_checked': len(manifest['files']),
        'supplementary_results_json_exact_match': True,
        'original_additional_metric_systems_checked': 5,
        'within_language_systems_checked': 6,
        'new_model_inference': False,
        'independent_human_validation': False,
        'manuscript_table_values_checked': False,
        'figure2_raster_match': None,
    }
    if a.manuscript:
        sources = '\n'.join(
            p.read_text() for p in (a.manuscript / 'chapters').glob('*.tex')
        )
        t = table(sources, 'tab:holdout-results')
        # Repeated system labels occur in the two panels: inspect each panel separately.
        panA = t.split('Panel A:')[1].split('Panel B:')[0]
        panB = t.split('Panel B:')[1].split('Panel C:')[0]
        for label, key in [
            ('H', 'H'),
            ('SV', 'SV'),
            ('SV-NLI', 'SV_NLI'),
            ('SV-NLI (repair)', 'SV_NLI_repair'),
            ('SV-NLI-base', 'SV_NLI_base'),
        ]:
            m = systems[key]
            printed(
                panA, label, [m[k] for k in ['spearman', 'kendall', 'pearson']], [3] * 3
            )
            printed(
                panB,
                label,
                [
                    m[k]
                    for k in ['mae', 'rmse', 'bias', 'extreme_over', 'extreme_under']
                ],
                [2] * 5,
            )
        panC = t.split('Panel C:')[1]
        for label, key in [
            ('SV minus H', 'SV minus H'),
            ('SV-NLI minus SV', 'SV_NLI minus SV'),
        ]:
            c = r['original_primary_multiplicity_sensitivity'][key]
            printed(
                panC,
                label,
                [c['estimate']] + c['nominal95'] + c['bonferroni_style97_5'],
                [3] * 5,
            )
        t = table(sources, 'tab:posthoc-controls')
        for label, key in [
            ('H', 'H'),
            ('Unconditional mean', 'Mean'),
            ('SV', 'SV'),
            ('CalH', 'CalH'),
        ]:
            m = systems[key]
            printed(
                t,
                label,
                [
                    m[k]
                    for k in [
                        'spearman',
                        'mae',
                        'bias',
                        'extreme_over',
                        'extreme_under',
                    ]
                ],
                [3, 2, 2, 2, 2],
            )
        t = table(sources, 'tab:within-language')
        for label, key in [
            ('H', 'H'),
            ('Mean', 'Mean'),
            ('SV', 'SV'),
            ('CalH', 'CalH'),
            ('SV-NLI', 'SV_NLI'),
            ('SV-NLI-base', 'SV_NLI_base'),
        ]:
            m = within[key]
            printed(
                t,
                label,
                [
                    m[k]
                    for k in ['defined', 'undefined', 'median', 'q1', 'q3', 'positive']
                ],
                [0, 0, 3, 3, 3, 0],
            )
        report['manuscript_table_values_checked'] = True
        # Render the regenerated PDF and the delivered vector PDF identically.
        import fitz

        generated = a.out / 'figures'
        subprocess.run(
            [
                sys.executable,
                str(root / 'figures/make_figures.py'),
                '--results',
                str(a.reproduced / 'RESULTS.json'),
                '--out',
                str(generated),
            ],
            check=True,
        )
        with fitz.open(generated / 'primary_contrasts.pdf') as pdf:
            one = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2)).samples
        with fitz.open(a.manuscript / 'figures/primary_contrasts.pdf') as pdf:
            two = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2)).samples
        assert one == two, 'Regenerated contrast plot differs at raster level.'
        report['figure2_raster_match'] = True
        report['stable_source_labels'] = [
            'tab:holdout-results',
            'tab:posthoc-controls',
            'tab:within-language',
            'fig:primary-contrasts',
        ]
    (a.out / 'VERIFICATION.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
