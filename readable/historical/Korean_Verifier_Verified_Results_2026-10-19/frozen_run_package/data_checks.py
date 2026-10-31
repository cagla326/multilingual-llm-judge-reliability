from pathlib import Path
import collections, hashlib, json, re

ROOT = Path(__file__).resolve().parent
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
H = lambda s: hashlib.sha256(norm(s).encode()).hexdigest()


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def validate_data():
    m = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    src = m['source']
    assert (
        hashlib.sha256((ROOT / src['file']).read_bytes()).hexdigest() == src['sha256']
    )
    originals = {r['guid']: r for r in json.loads((ROOT / src['file']).read_text())}
    previous = json.loads(
        (ROOT / 'data/PREVIOUSLY_INSPECTED_PUBLIC_DEV.json').read_text()
    )
    assert (
        hashlib.sha256(
            (ROOT / 'data/PREVIOUSLY_INSPECTED_PUBLIC_DEV.json').read_bytes()
        ).hexdigest()
        == m['previously_inspected_public_dev_sha256']
    )
    blocked = {H(r[k]) for r in previous for k in ['premise', 'hypothesis']}
    all_ids = set()
    splittexts = {}
    splitpremises = {}
    result = {}
    for split, n in [('train', 3000), ('validation', 600), ('internal_test', 600)]:
        rows = read_rows(ROOT / 'data' / f'{split}.jsonl')
        assert len(rows) == len({r['key'] for r in rows}) == n
        assert not all_ids & {r['key'] for r in rows}
        all_ids |= {r['key'] for r in rows}
        groups = collections.defaultdict(list)
        for r in rows:
            original = originals[r['key']]
            assert (
                r['premise'] == original['premise']
                and r['hypothesis'] == original['hypothesis']
                and r['human_label'] == original['gold_label']
            )
            assert r['genre'] == original['genre']
            assert r['premise_hash'] == H(r['premise'])
            assert [
                original[k] for k in ['author', 'label2', 'label3', 'label4', 'label5']
            ].count(original['gold_label']) >= 3
            assert r['answer'] == m['label_mapping'][r['human_label']]
            assert r['prompt'] == m['prompt_template'].format(
                premise=r['premise'], hypothesis=r['hypothesis']
            )
            groups[r['premise_hash']].append(r)
        for rr in groups.values():
            assert collections.Counter(r['human_label'] for r in rr) == {
                'entailment': 1,
                'contradiction': 1,
                'neutral': 1,
            }
        splittexts[split] = {H(r[k]) for r in rows for k in ['premise', 'hypothesis']}
        splitpremises[split] = set(groups)
        assert not splittexts[split] & blocked
        result[split] = {
            'rows': n,
            'premises': len(groups),
            'labels': dict(collections.Counter(r['human_label'] for r in rows)),
        }
    for a, b in [
        ('train', 'validation'),
        ('train', 'internal_test'),
        ('validation', 'internal_test'),
    ]:
        assert (
            not splittexts[a] & splittexts[b]
            and not splitpremises[a] & splitpremises[b]
        )
    return result


if __name__ == '__main__':
    print(json.dumps(validate_data(), indent=2))
