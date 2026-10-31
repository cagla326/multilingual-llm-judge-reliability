#!/usr/bin/env python3
"""Human-labelled data only. Source- and exact-text-disjoint internal development splits."""
from pathlib import Path
import collections, csv, hashlib, json, random, re, shutil

ROOT = Path(__file__).resolve().parent
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
H = lambda s: hashlib.sha256(norm(s).encode()).hexdigest()


def main():
    src = json.loads((ROOT / 'TRAIN_SOURCE.json').read_text())
    path = ROOT / src['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == src['sha256']
    raw = json.loads(path.read_text())
    groups = collections.defaultdict(list)
    for r in raw:
        assert [r[k] for k in ['author', 'label2', 'label3', 'label4', 'label5']].count(
            r['gold_label']
        ) >= 3
        groups[H(r['premise'])].append(r)
    old = ROOT.parent / 'KLUE_NLI_Transfer_2026-10-19'
    public = json.loads((old / 'data/klue-nli-v1.1_dev.json').read_text())
    blocked_text = {H(r[k]) for r in public for k in ['premise', 'hypothesis']}
    hist = ROOT.parent / 'korean_evidence_2026-10-19/korean_records'
    prior = []
    prior_meta = []
    for name in [
        'korean_xlsum_TRAIN50_sources.csv',
        'korean_xlsum_FROZEN_TEST50_sources.csv',
        'korean_xlsum_train_pool_50.csv',
    ]:
        p = hist / name
        with p.open() as f:
            rr = list(csv.DictReader(f))
        prior += [norm(r['article']) for r in rr]
        prior_meta.append(
            {
                'file': name,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'rows': len(rr),
            }
        )
    p = hist / 'korean_train_5k.jsonl'
    rr = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    prior += [norm(r['document']) for r in rr]
    prior_meta.append(
        {
            'file': p.name,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'rows': len(rr),
        }
    )
    p = ROOT.parent / 'cluster_evidence_2026-10-19/fullsource_rerun_input_frozen.csv'
    with p.open() as f:
        rr = list(csv.DictReader(f))
    prior += [norm(r['input']) for r in rr]
    prior_meta.append(
        {
            'file': p.name,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'rows': len(rr),
        }
    )
    eligible = collections.defaultdict(list)
    rejected = collections.Counter()
    for ph, rr in sorted(groups.items()):
        if len(rr) != 3 or collections.Counter(r['gold_label'] for r in rr) != {
            'entailment': 1,
            'contradiction': 1,
            'neutral': 1,
        }:
            rejected['not_exactly_one_of_each_class'] += 1
            continue
        if len({r['premise'] for r in rr}) != 1 or len({r['genre'] for r in rr}) != 1:
            rejected['source_metadata_disagreement'] += 1
            continue
        text_ids = {H(rr[0]['premise'])} | {H(r['hypothesis']) for r in rr}
        if text_ids & blocked_text:
            rejected['exact_text_overlap_with_previously_inspected_public_dev'] += 1
            continue
        pr = norm(rr[0]['premise'])
        if any(pr in t or t in pr for t in prior):
            rejected['prior_thesis_source_substring_overlap'] += 1
            continue
        eligible[rr[0]['genre']].append((ph, rr, text_ids))
    rng = random.Random(20261019)
    for genre in sorted(eligible):
        rng.shuffle(eligible[genre])
    counts = {
        'internal_test': {
            'NSMC': 40,
            'airbnb': 40,
            'policy': 30,
            'wikinews': 30,
            'wikipedia': 30,
            'wikitree': 30,
        },
        'validation': {
            'NSMC': 40,
            'airbnb': 40,
            'policy': 30,
            'wikinews': 30,
            'wikipedia': 30,
            'wikitree': 30,
        },
        'train': {
            'NSMC': 200,
            'airbnb': 200,
            'policy': 150,
            'wikinews': 150,
            'wikipedia': 150,
            'wikitree': 150,
        },
    }
    labels = {'entailment': 'A', 'contradiction': 'B', 'neutral': 'C'}
    template = '근거 문장과 판단할 문장을 읽고 의미 관계를 분류하세요.\n\n근거 문장:\n{premise}\n\n판단할 문장:\n{hypothesis}\n\nA = 근거 문장이 참이면 판단할 문장도 반드시 참이다.\nB = 근거 문장이 참이면 판단할 문장은 반드시 거짓이다.\nC = 근거 문장만으로 판단할 문장의 참과 거짓을 결정할 수 없다.\n\nA, B, C 중 하나만 답하세요.'
    used_premises = set()
    used_text = set()
    assign = {}
    fullrefs = {}
    for split, quota in counts.items():
        selected = []
        this_text = set()
        for genre, n in quota.items():
            nfound = 0
            for ph, rr, txt in eligible[genre]:
                if ph in used_premises or txt & used_text:
                    continue
                selected += rr
                this_text |= txt
                used_premises.add(ph)
                nfound += 1
                if nfound == n:
                    break
            assert nfound == n, (split, genre, nfound, n)
        used_text |= this_text
        assign[split] = []
        for r in sorted(selected, key=lambda r: r['guid']):
            fullrefs[r['guid']] = r
            assign[split].append(
                {
                    'key': r['guid'],
                    'premise_hash': H(r['premise']),
                    'genre': r['genre'],
                    'premise': r['premise'],
                    'hypothesis': r['hypothesis'],
                    'human_label': r['gold_label'],
                    'answer': labels[r['gold_label']],
                    'prompt': template.format(
                        premise=r['premise'], hypothesis=r['hypothesis']
                    ),
                }
            )
        (ROOT / 'data' / f'{split}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in assign[split])
        )
    (ROOT / 'data/selected_original_records.json').write_text(
        json.dumps(fullrefs, ensure_ascii=False)
    )
    for name in ['UPSTREAM_License.md', 'UPSTREAM_README.md']:
        shutil.copy2(old / 'data' / name, ROOT / 'data' / name)
    shutil.copy2(
        old / 'data/klue-nli-v1.1_dev.json',
        ROOT / 'data/PREVIOUSLY_INSPECTED_PUBLIC_DEV.json',
    )
    manifest = {
        'status': 'PREPARED_NO_TRAINING_OR_MODEL_OUTPUTS',
        'seed': 20261019,
        'source': src,
        'previously_inspected_public_dev_sha256': hashlib.sha256(
            (ROOT / 'data/PREVIOUSLY_INSPECTED_PUBLIC_DEV.json').read_bytes()
        ).hexdigest(),
        'raw_rows': len(raw),
        'raw_normalized_premise_groups': len(groups),
        'eligible_by_genre': {g: len(v) for g, v in eligible.items()},
        'rejected_group_counts': dict(rejected),
        'split_counts': {
            s: {
                'rows': len(rr),
                'premises': len({r['premise_hash'] for r in rr}),
                'labels': dict(collections.Counter(r['human_label'] for r in rr)),
            }
            for s, rr in assign.items()
        },
        'selection_order': ['internal_test', 'validation', 'train'],
        'source_quotas': counts,
        'text_disjointness': 'No exact normalized premise or hypothesis text is shared across internal splits or with the inspected public dev. Within-split candidate repetition is allowed. Not semantic deduplication.',
        'prior_source_checks': prior_meta,
        'prompt_template': template,
        'label_mapping': labels,
        'new_example_text': False,
        'new_human_label': False,
    }
    (ROOT / 'DATA_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest['split_counts'], indent=2))


if __name__ == '__main__':
    main()
