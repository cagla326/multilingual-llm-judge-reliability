"""Post-hoc saved-score controls for the trained verifier (SV-NLI).

Step 2: selective vs unconditional combination with the trained verifier.
    SV_NLI   = selective(H, c_NLI)        (existing system, same 75/50 trigger)
    Mean_NLI = (H + c_NLI) / 2            (every example, same claim scores and
                                           empty-claim rule, since c_NLI is reused)
Step 3: development-only additive calibration of SV_NLI and Mean_NLI.
    offset_s = mean_dev(y - s);  Cal_s = clip(s + offset_s, 0, 100)
    The offset is fitted only on the 900 development rows and applied unchanged
    to the additional sample. No additional-sample label enters any fit.

No GPU inference. Same language-cluster percentile bootstrap as
analyze_controls.py (45 languages, 2,000 draws), with its own fixed seed.
Write into a NEW output directory.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

SEED = 20261024
DRAWS = 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selective(h, c):
    return np.where(((h >= 75) & (c <= 50)) | ((h <= 50) & (c >= 75)), (h + c) / 2, h)


def rho(y, s):
    if np.unique(y).size > 1 and np.unique(s).size > 1:
        return float(spearmanr(y, s).statistic)
    return None


def metrics(y, s):
    e = s - y
    return dict(
        n=int(len(y)),
        spearman=rho(y, s),
        mae=float(np.abs(e).mean()),
        rmse=float(np.sqrt(np.mean(e * e))),
        bias=float(e.mean()),
        extreme_over=float(100 * np.mean(e >= 50)),
        extreme_under=float(100 * np.mean(e <= -50)),
    )


def bootstrap(df, pairs, seed=SEED, draws=DRAWS):
    rng = np.random.default_rng(seed)
    langs = sorted(df.lang.unique())
    groups = [np.flatnonzero(df.lang.to_numpy() == l) for l in langs]
    y = df.y.to_numpy()
    cols = sorted({s for p in pairs for s in p})
    sc = {s: df[s].to_numpy() for s in cols}
    keys = ['spearman', 'mae', 'extreme_over', 'extreme_under']
    out = {f'{a} minus {b}': {k: [] for k in keys} for a, b in pairs}
    for _ in range(draws):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        m = {s: metrics(y[ix], v[ix]) for s, v in sc.items()}
        for a, b in pairs:
            for k in keys:
                out[f'{a} minus {b}'][k].append(m[a][k] - m[b][k])
    return {
        name: {k: [float(x) for x in np.percentile(v, (2.5, 97.5))] for k, v in d.items()}
        for name, d in out.items()
    }


def ties(s):
    """Number of tied pairs among scores (Spearman uses average ranks for ties)."""
    _, counts = np.unique(s, return_counts=True)
    return int(sum(c * (c - 1) // 2 for c in counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--additional-rows', type=Path, required=True,
                    help='directory with the saved row_*.json files of the additional run')
    ap.add_argument('--additional-labels', type=Path, required=True,
                    help='sealed additional labels CSV (holdout_row, id, q2_avg)')
    ap.add_argument('--development', type=Path, required=True,
                    help='supplementary_controls/data/development900.csv from the evidence archive')
    ap.add_argument('--bridge-rows', type=Path, required=True,
                    help='results_v1/FINAL_ROWS.jsonl from archival/Trigger_Bridge_Verified_Results_2026-10-20.zip')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'RESULTS.json').exists():
        raise RuntimeError('Choose a new --out directory.')

    labels = pd.read_csv(a.additional_labels).set_index('holdout_row'); assert labels.index.is_unique
    rec = []
    for p in sorted(a.additional_rows.glob('row_*.json')):
        r = json.loads(p.read_text())
        k = int(r['holdout_row'])
        assert int(r['id']) == int(labels.loc[k, 'id'])
        h = float(r['H']['score'])
        c = float(r['NLI_T']['trained3']['capped'])
        rec.append(dict(row_id=k, lang=r['lang'], y=float(labels.loc[k, 'q2_avg']) * 100,
                        H=h, c_NLI=c, SV_NLI=float(selective(h, c)), Mean_NLI=(h + c) / 2))
    add = pd.DataFrame(rec).sort_values('row_id').reset_index(drop=True)
    assert len(add) == 900 and add.groupby('lang').size().eq(20).all()

    systems = ['H', 'SV_NLI', 'Mean_NLI']
    pairs = [('Mean_NLI', 'SV_NLI'), ('Mean_NLI', 'H')]
    calib = None
    if True:
        d900 = pd.read_csv(a.development)
        br = pd.read_json(a.bridge_rows, lines=True)
        br['row_id'] = br.row_id.astype(int)
        m = d900.merge(br[['row_id', 'holistic', 'trained3_capped', 'trained3_selective']],
                       on='row_id', validate='one_to_one')
        assert len(m) == 900 and np.allclose(m.holistic, m.gemma_fullsource_score)
        dev = pd.DataFrame(dict(lang=m.lang, y=m.human_0_100, H=m.gemma_fullsource_score,
                                c_nli=m.trained3_capped))
        assert np.allclose(selective(dev.H, dev.c_nli), m.trained3_selective)
        dev['SV_NLI'] = selective(dev.H, dev.c_nli)
        dev['Mean_NLI'] = (dev.H + dev.c_nli) / 2
        calib = {}
        for s in ['SV_NLI', 'Mean_NLI']:
            off = float((dev.y - dev[s]).mean())           # development labels only
            raw = add[s] + off
            cal = np.clip(raw, 0, 100)
            add['Cal_' + s] = cal
            calib[s] = dict(offset=off, fit_n=900, fit_input_sha256={'development': sha(a.development), 'bridge_rows': sha(a.bridge_rows)},
                            clipped_low=int((raw < 0).sum()), clipped_high=int((raw > 100).sum()),
                            tied_pairs_before=ties(add[s].to_numpy()),
                            tied_pairs_after=ties(cal.to_numpy()))
        systems += ['Cal_SV_NLI', 'Cal_Mean_NLI']
        pairs += [('Cal_SV_NLI', 'SV_NLI'), ('Cal_Mean_NLI', 'Mean_NLI'), ('Cal_Mean_NLI', 'Cal_SV_NLI')]

    overall = {s: metrics(add.y.to_numpy(), add[s].to_numpy()) for s in systems}
    ci = bootstrap(add, pairs)
    contrasts = {
        f'{x} minus {y}': {'estimate': {k: overall[x][k] - overall[y][k]
                                        for k in ['spearman', 'mae', 'extreme_over', 'extreme_under']},
                           'ci95': ci[f'{x} minus {y}']}
        for x, y in pairs
    }
    result = dict(status='POST_HOC_TRAINED_VERIFIER_CONTROLS', seed=SEED, draws=DRAWS,
                  inputs={'additional_labels_sha256': sha(a.additional_labels), 'development_sha256': sha(a.development), 'bridge_rows_sha256': sha(a.bridge_rows)},
                  overall=overall, contrasts=contrasts, calibration=calib,
                  triggered_rows=int((add.SV_NLI != add.H).sum()),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   pandas=pd.__version__, scipy=scipy.__version__))
    (a.out / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    add.to_csv(a.out / 'scores.csv', index=False)
    print(json.dumps({k: result[k] for k in ['overall', 'contrasts', 'calibration']}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
