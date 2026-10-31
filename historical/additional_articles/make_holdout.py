"""Draw the untouched confirmatory sample (no model output, no look at human scores).

Exclusion: every (language, doc_id) that appears in either development sample
(the 900-row Q2 benchmark and the 900-row 'atomic' sample) is excluded with all
seven of its system summaries. From the remaining articles, 20 articles per
language are drawn with a fixed seed, and one system summary per article is drawn
uniformly at random (at most one row per article).
Outputs: inputs without human scores (for inference) and a sealed label file.
"""
import hashlib, json, re, sys
import numpy as np, pandas as pd

FULL, DEV900, ATOMIC = sys.argv[1:4]
SEED, PER_LANG = 20261020, 20
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

d = pd.read_csv(FULL)
dev = pd.read_csv(DEV900)
atomic = pd.read_csv(ATOMIC)
assert len(d) == 31500 and len(dev) == 900 and len(atomic) == 900

key = lambda x: x.lang + "||" + x.system + "||" + x.summary.map(norm) + "||" + x.input.map(norm)
d["_k"] = key(d)
dev_rows = d[d._k.isin(set(key(dev)))]
assert len(dev_rows) == 900, "development rows must all be found"
atomic_rows = d[d.id.isin(set(atomic.id))]
assert len(atomic_rows) == 900
excluded = set(zip(dev_rows.lang, dev_rows.doc_id)) | set(zip(atomic_rows.lang, atomic_rows.doc_id))

d["_seen"] = [(l, i) in excluded for l, i in zip(d.lang, d.doc_id)]
pool = d[~d._seen]
rng = np.random.default_rng(SEED)
picked = []
for lang in sorted(pool.lang.unique()):
    docs = sorted(pool.loc[pool.lang == lang, "doc_id"].unique())
    chosen = rng.choice(docs, PER_LANG, replace=False)
    for doc in sorted(chosen):
        cands = pool[(pool.lang == lang) & (pool.doc_id == doc)].sort_values("system")
        picked.append(cands.iloc[int(rng.integers(len(cands)))])
h = pd.DataFrame(picked).reset_index(drop=True)
h.insert(0, "holdout_row", range(len(h)))
assert len(h) == 900 and h.groupby("lang").size().eq(PER_LANG).all()
assert not set(zip(h.lang, h.doc_id)) & excluded
assert h.doc_id.groupby(h.lang).nunique().eq(PER_LANG).all()

inputs = h[["holdout_row", "id", "doc_id", "lang", "split", "system", "input", "summary"]]
labels = h[["holdout_row", "id", "q1_min", "q1_avg", "q1_max", "q2_min", "q2_avg", "q2_max",
            "q3_min", "q3_avg", "q3_max"]]
inputs.to_csv("holdout900_inputs_NO_HUMAN_SCORES.csv", index=False)
labels.to_csv("holdout900_labels_SEALED.csv", index=False)
pd.DataFrame(sorted(excluded), columns=["lang", "doc_id"]).to_csv("excluded_development_articles.csv", index=False)
manifest = {
    "seed": SEED, "per_language": PER_LANG, "rows": len(h),
    "excluded_articles": len(excluded), "eligible_articles": int(pool.drop_duplicates(["lang", "doc_id"]).shape[0]),
    "source_sha256": {"full": sha(FULL), "dev900": sha(DEV900), "atomic": sha(ATOMIC)},
    "output_sha256": {f: sha(f) for f in ["holdout900_inputs_NO_HUMAN_SCORES.csv",
                                           "holdout900_labels_SEALED.csv", "excluded_development_articles.csv"]},
    "system_counts": h.system.value_counts().sort_index().to_dict(),
    "human_scores_inspected": False,
}
json.dump(manifest, open("HOLDOUT_MANIFEST.json", "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k != "output_sha256"}, indent=2))
