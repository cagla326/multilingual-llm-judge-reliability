"""Exploratory comparator: the NLI-based factuality score released with mFACE.

Added 30 October 2026, after all reported results had been inspected. No model
is run and no reported score changes. The released score (column
``e2e_nli_score``) is a fixed metric distributed with the benchmark; nothing is
fitted. Because it is on a different scale from the 0-100 judge scores, only
rank-based measures are compared (Spearman, Kendall tau-b).

The full release is not bundled with the archive (licence and size). Pass the
release CSV with --release. The script does not require a byte-identical copy:
it checks that every selected development and additional row matches the
archived identifiers, human scores and (for development) the archived released
NLI scores, and records the SHA-256 of the file it was given.

Contrasts use the same language-cluster bootstrap (2,000 draws) and the same
seed as the other post-hoc controls (20261024). Resampled language indices do
not depend on which systems are compared, so the draws are aligned with the
existing control intervals.

Usage: python supplementary_controls/released_nli_comparator.py \
           --release <mface_annotated_data.csv> --out <new dir> [--evidence-root <archive root>]
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

SEED = 20261024
DRAWS = 2000


def rho(y, s):
    return float(spearmanr(y, s).statistic) if np.unique(y).size > 1 and np.unique(s).size > 1 else None


def selective(h, c):
    return np.where(((h >= 75) & (c <= 50)) | ((h <= 50) & (c >= 75)), (h + c) / 2, h)


def bootstrap(df, pairs):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(df.lang.to_numpy() == l) for l in sorted(df.lang.unique())]
    y = df.y.to_numpy()
    out = {f"{a} minus {b}": [] for a, b in pairs}
    names = sorted({s for p in pairs for s in p})
    for _ in range(DRAWS):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        r = {s: rho(y[ix], df[s].to_numpy()[ix]) for s in names}
        for a, b in pairs:
            out[f"{a} minus {b}"].append(r[a] - r[b])
    return out


def summarise(df, systems, pairs):
    y = df.y.to_numpy()
    overall = {s: {"spearman": rho(y, df[s].to_numpy()),
                   "kendall": float(kendalltau(y, df[s].to_numpy()).statistic)} for s in systems}
    draws = bootstrap(df, pairs)
    contrasts = {k: {"estimate": overall[k.split(" minus ")[0]]["spearman"] - overall[k.split(" minus ")[1]]["spearman"],
                     "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]} for k, v in draws.items()}
    by_system = {}
    for name, g in df.groupby("system"):
        by_system[name] = {"n": int(len(g)), "spearman_released_nli": rho(g.y.to_numpy(), g.released_nli.to_numpy())}
    vals = [v["spearman_released_nli"] for v in by_system.values() if v["spearman_released_nli"] is not None]
    within_lang = [rho(g.y.to_numpy(), g.released_nli.to_numpy()) for _, g in df.groupby("lang")]
    wl = np.array([v for v in within_lang if v is not None])
    return {"n": int(len(df)), "overall": overall, "contrasts_spearman": contrasts,
            "within_system_spearman_range": [min(vals), max(vals)], "by_system": by_system,
            "within_language_released_nli": {"defined": int(len(wl)), "median": float(np.median(wl)),
                                             "q1": float(np.percentile(wl, 25)), "q3": float(np.percentile(wl, 75))}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-root", default=None)
    a = ap.parse_args()
    root = Path(a.evidence_root).resolve() if a.evidence_root else Path(__file__).resolve().parents[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rel_path = Path(a.release)
    rel = pd.read_csv(rel_path)
    assert len(rel) == 31500 and rel.id.is_unique

    ids = pd.read_csv(root / "data/selected_article_ids.csv")
    rel_i = rel.set_index("id")

    # Development: archived input carries the released score; it must match exactly.
    devin = pd.read_csv(root / "data/development_frozen_input.csv")
    for c in ["doc_id", "lang", "system", "q2_avg", "e2e_nli_score"]:
        a_, b_ = devin[c].to_numpy(), rel_i.loc[devin.id, c].to_numpy()
        assert (np.isclose(a_, b_) if c in ("q2_avg", "e2e_nli_score") else (a_ == b_)).all(), c
    # The scored development file has no release id; join on (lang, system, summary), which is one-to-one.
    devsc = pd.read_csv(root / "supplementary_controls/data/development900.csv")
    devsc = devsc.merge(devin[["id", "lang", "system", "summary", "input", "q2_avg", "e2e_nli_score"]],
                        on=["lang", "system", "summary"], how="left", validate="one_to_one")
    assert devsc.id.notna().all() and (devsc.input_x == devsc.input_y).all()
    assert np.allclose(devsc.human_0_100, devsc.q2_avg * 100)
    dev = pd.DataFrame({"lang": devsc.lang, "system": devsc.system, "y": devsc.human_0_100,
                        "H": devsc.gemma_fullsource_score,
                        "SV": selective(devsc.gemma_fullsource_score.to_numpy(), devsc.claim_capped_score.to_numpy()),
                        "released_nli": devsc.e2e_nli_score})

    # Additional: archived scores plus trained-verifier claim scores from the row records.
    add = pd.read_csv(root / "supplementary_controls/results_v1/additional_scores.csv")
    sel = ids[ids["sample"] == "additional"].set_index("id")
    assert set(add.id) == set(sel.index)
    r = rel_i.loc[add.id]
    assert (r.lang.to_numpy() == add.lang.to_numpy()).all()
    assert (r.doc_id.to_numpy() == sel.loc[add.id, "doc_id"].to_numpy()).all()
    assert np.allclose(r.q2_avg.to_numpy() * 100, add.y.to_numpy())
    nli_claim = {}
    for p in sorted((root / "archival/additional_run/run_v1/rows").glob("row_*.json")):
        rec = json.loads(p.read_text())
        nli_claim[int(rec["holdout_row"])] = float(rec["NLI_T"]["trained3"]["capped"])
    add["Mean_NLI"] = (add.H + add.row_id.map(nli_claim)) / 2
    add["system"] = r.system.to_numpy()
    add["released_nli"] = r.e2e_nli_score.to_numpy()

    res = {"status": "EXPLORATORY_POST_HOC", "added": "2026-10-30",
           "release_file_sha256": hashlib.sha256(rel_path.read_bytes()).hexdigest(),
           "release_check": "selected rows match archived identifiers, human scores and development released-NLI values",
           "seed": SEED, "draws": DRAWS,
           "development": summarise(dev, ["H", "SV", "released_nli"],
                                    [("released_nli", "H"), ("released_nli", "SV")]),
           "additional": summarise(add, ["H", "SV", "Mean", "SV_NLI", "Mean_NLI", "released_nli"],
                                   [("released_nli", "H"), ("released_nli", "SV"),
                                    ("released_nli", "SV_NLI"), ("released_nli", "Mean_NLI")]),
           "limits": ["Specified after all reported results had been inspected; exploratory.",
                      "Released score is on a different scale; only rank-based measures are compared.",
                      "Some mFACE systems were trained or selected with NLI signals, which may favour this score.",
                      "Intervals are nominal language-cluster bootstrap intervals, not adjusted for multiplicity."]}
    (out / "RESULTS.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ["development", "additional"]}, indent=1)[:6000])
