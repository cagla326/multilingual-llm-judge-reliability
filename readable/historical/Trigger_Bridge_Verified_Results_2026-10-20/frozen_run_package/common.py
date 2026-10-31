import hashlib, json, csv, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ['entailment', 'contradiction', 'neutral']
REV = '093f9f388b31de276ce2de164bdc2081324b9767'
CONFIG = 'cde99458eaf1fceff52cad78e46fe868eb3bfc268d4b9d54dd6b9ea01a457334'
WEIGHTS = '818a9c918109536d0c3fd90598303030f0ffc5d3885b986f7153b65f95d2472d'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(
        json.dumps(x, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))


def capped(labels):
    if not labels:
        return 0.0
    if any(x not in CLASSES for x in labels):
        raise ValueError('Unknown NLI label')
    p = 100 * labels.count('entailment') / len(labels)
    return (
        min(p, 25)
        if 'contradiction' in labels
        else min(p, 50) if 'neutral' in labels else p
    )


def trigger(b, c):
    return (b + c) / 2 if (b >= 75 and c <= 50) or (b <= 50 and c >= 75) else b


def validate():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    for n, h in protocol['package_hashes'].items():
        if sha(ROOT / n) != h:
            raise ValueError('Package changed: ' + n)
    meta = rows(ROOT / 'data/examples.jsonl')
    claims = rows(ROOT / 'data/claims.jsonl')
    byid = {r['row_id']: r for r in meta}
    byclaim = {r['key']: r for r in claims}
    assert len(meta) == len(byid) == 900 and len(claims) == len(byclaim) == 2489
    assert len({r['lang'] for r in meta}) == 45
    from collections import Counter

    assert set(Counter(r['lang'] for r in meta).values()) == {20}
    with (ROOT / 'data/original_final900.csv').open() as f:
        original = list(csv.DictReader(f))
    manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    template = manifest['prompt_template']
    for r in original:
        ex = byid[r['row_id']]
        cs = json.loads(r['claims'])
        assert ex['claim_keys'] == [f"{r['row_id']}:{i}" for i in range(len(cs))]
        assert ex['lang'] == r['lang'] and ex['system'] == r['system']
        assert (
            ex['human'] == float(r['human_0_100'])
            and ex['baseline'] == float(r['gemma_fullsource_score'])
            and ex['original_final'] == float(r['final_symmetric_extreme_average'])
        )
        assert math.isclose(
            trigger(ex['baseline'], float(r['claim_capped_score'])),
            ex['original_final'],
            abs_tol=1e-8,
        )
        for i, claim in enumerate(cs):
            cr = byclaim[f"{r['row_id']}:{i}"]
            assert (
                cr['premise'] == r['translated_source_en'] and cr['hypothesis'] == claim
            )
            assert (
                cr['prompt']
                == template.format(premise=cr['premise'], hypothesis=cr['hypothesis'])
                and cr['row_id'] == r['row_id']
            )
            assert cr['premise'].strip() and cr['hypothesis'].strip()
    return protocol, meta, claims
