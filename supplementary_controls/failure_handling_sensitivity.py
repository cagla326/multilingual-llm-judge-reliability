"""Exploratory sensitivity check: does the failure-as-zero policy drive the results?

Added 28 October 2026, after all reported results had been inspected. This is
an exploratory saved-score analysis. It runs no model and changes no reported
system, threshold, endpoint or score.

Reported policy (as frozen in the original pipelines)
    * an extraction output that cannot be parsed, or a valid but empty claim
      list, gives a claim score of 0;
    * a four-label verification response that cannot be parsed counts as
      `unsupported`.

Alternative policies examined here
    exclude   drop every row affected by any such failure, using one common
              row set for all systems so that contrasts stay paired;
    fallback  rows without claims take the holistic score as their claim
              score (so the selective rule leaves H unchanged), and failed
              four-label verifications are dropped before aggregation (if no
              claim remains, the claim score falls back to H as well).

The trained NLI verifier reads restricted next-token logits and therefore
has no verification parse failures; it is affected only through empty or
unparsable claim lists.

Inputs: the 900 additional-sample row records, their human labels, and the
900-row development output. Same language-cluster percentile bootstrap as the
original analyses (45 languages, 2,000 draws), with the seeds those analyses
used: 20261021 for the two primary contrasts and 20261024 for the post-hoc
control contrasts, so that the reported column reproduces the thesis values.

Usage:  python supplementary_controls/failure_handling_sensitivity.py --out <new dir>
        [--evidence-root <extracted evidence archive>]   (default: this archive)
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def set_root(root):
    global ROOT, ROWS, LABELS, DEV
    ROOT = Path(root).resolve()
    ROWS = ROOT / "archival/additional_run/run_v1/rows"
    LABELS = ROOT / "supplementary_controls/data/additional_labels.csv"
    DEV = ROOT / "supplementary_controls/data/development900.csv"


set_root(ROOT)
PRIMARY_SEED, CONTROL_SEED, DRAWS = 20261021, 20261024, 2000
CAP = {"fully_supported": 1.0, "partially_supported": 0.5}


def selective(h, c):
    return np.where(((h >= 75) & (c <= 50)) | ((h <= 50) & (c >= 75)), (h + c) / 2, h)


def capped(labels):
    labels = [str(x).strip().lower() for x in labels]
    if not labels:
        return None
    p = 100 * sum(CAP.get(x, 0.0) for x in labels) / len(labels)
    if "contradicted" in labels:
        return min(p, 25.0)
    if "unsupported" in labels:
        return min(p, 50.0)
    if "partially_supported" in labels:
        return min(p, 75.0)
    return p


def metrics(s, y):
    e = s - y
    return {"spearman": float(spearmanr(s, y)[0]), "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()), "extreme_over": float((e >= 50).mean() * 100),
            "extreme_under": float((e <= -50).mean() * 100)}


def contrasts(langs, y, systems, pairs, seed):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(langs))
    idx = {l: np.where(langs == l)[0] for l in uniq}
    keys = ["spearman", "mae", "extreme_under"]
    draws = {p: {m: [] for m in keys} for p in pairs}
    for _ in range(DRAWS):
        rows = np.concatenate([idx[l] for l in rng.choice(uniq, len(uniq), replace=True)])
        cache = {k: metrics(systems[k][rows], y[rows]) for k in {x for p in pairs for x in p}}
        for a, b in pairs:
            for m in keys:
                draws[(a, b)][m].append(cache[a][m] - cache[b][m])
    out = {}
    for a, b in pairs:
        pa, pb = metrics(systems[a], y), metrics(systems[b], y)
        out[f"{a} - {b}"] = {m: {"estimate": round(pa[m] - pb[m], 6),
                                 "ci95": [round(float(np.percentile(draws[(a, b)][m], 2.5)), 6),
                                          round(float(np.percentile(draws[(a, b)][m], 97.5)), 6)]}
                             for m in keys}
    return out


def additional():
    lab = pd.read_csv(LABELS)
    y_map = dict(zip(lab.holdout_row.astype(str), lab.q2_avg.astype(float) * 100))
    recs = [json.loads(p.read_text()) for p in sorted(ROWS.glob("row_*.json"))]
    recs.sort(key=lambda r: int(r["holdout_row"]))
    assert len(recs) == 900
    out = {"langs": [], "y": [], "H": [], "c0": [], "c0_fb": [], "cN": [], "cN_fb": [], "fail": []}
    counts = {"extraction_unparsable": 0, "empty_claim_list": 0,
              "rows_with_verifier_parse_failure": 0, "verifier_parse_failures": 0}
    for r in recs:
        h = float(r["H"]["score"])
        claims = r["T"]["claims"]
        ext_bad = not r["T"]["claims_json_valid"]
        empty = len(claims) == 0
        vf = [v for v in r["S0"]["verifications"] if not v["valid_json"]]
        counts["extraction_unparsable"] += int(ext_bad)
        counts["empty_claim_list"] += int(empty and not ext_bad)
        counts["rows_with_verifier_parse_failure"] += int(bool(vf))
        counts["verifier_parse_failures"] += len(vf)
        # reported claim scores (reproduced, then checked against the record)
        c0 = capped(r["S0"]["labels"])
        c0 = 0.0 if c0 is None else c0
        assert abs(c0 - float(r["S0"]["capped"])) < 1e-9
        cN = float(r["NLI_T"]["trained3"]["capped"])
        # fallback policy
        kept = [v["label"] for v in r["S0"]["verifications"] if v["valid_json"]]
        c0_fb = capped(kept)
        c0_fb = h if (empty or c0_fb is None) else c0_fb
        cN_fb = h if empty else cN
        out["langs"].append(r["lang"]); out["y"].append(y_map[r["holdout_row"]]); out["H"].append(h)
        out["c0"].append(c0); out["c0_fb"].append(c0_fb); out["cN"].append(cN); out["cN_fb"].append(cN_fb)
        out["fail"].append(bool(ext_bad or empty or vf))
    a = {k: np.array(v) for k, v in out.items()}
    H = a["H"]
    systems = {
        "reported": {"H": H, "SV": selective(H, a["c0"]), "Mean": (H + a["c0"]) / 2,
                     "SV-NLI": selective(H, a["cN"]), "Mean-NLI": (H + a["cN"]) / 2},
        "fallback": {"H": H, "SV": selective(H, a["c0_fb"]), "Mean": (H + a["c0_fb"]) / 2,
                     "SV-NLI": selective(H, a["cN_fb"]), "Mean-NLI": (H + a["cN_fb"]) / 2},
    }
    keep = ~a["fail"]
    systems["exclude"] = {k: v[keep] for k, v in systems["reported"].items()}
    primary = [("SV", "H"), ("SV-NLI", "SV")]
    controls = [("Mean", "SV"), ("Mean-NLI", "SV-NLI")]
    res = {"counts": {**counts, "rows_affected_any": int(a["fail"].sum())}}
    for pol, sysd in systems.items():
        y = a["y"][keep] if pol == "exclude" else a["y"]
        langs = a["langs"][keep] if pol == "exclude" else a["langs"]
        res[pol] = {"n_rows": int(len(y)),
                    "systems": {k: {m: round(x, 6) for m, x in metrics(v, y).items()} for k, v in sysd.items()},
                                        "contrasts": {**contrasts(langs, y, sysd, primary, PRIMARY_SEED),
                                  **contrasts(langs, y, sysd, controls, CONTROL_SEED)}}
    return res


def development():
    d = pd.read_csv(DEV)
    y = d.human_0_100.to_numpy(float)
    H = d.gemma_fullsource_score.to_numpy(float)
    langs = d.lang.to_numpy()
    c0, c0_fb, fail = [], [], []
    for _, r in d.iterrows():
        ver = json.loads(r.claim_verifications) if isinstance(r.claim_verifications, str) else []
        labels = [v.get("label", "unsupported") for v in ver]
        empty = len(ver) == 0
        bad = [v for v in ver if not v.get("valid_json", True)]
        c = capped(labels); c = 0.0 if c is None else c
        assert abs(c - float(r.claim_capped_score)) < 1e-9, r.row_id
        kept = [v.get("label", "unsupported") for v in ver if v.get("valid_json", True)]
        cf = capped(kept)
        c0.append(c); c0_fb.append(r.gemma_fullsource_score if (empty or cf is None) else cf)
        fail.append(bool(empty or bad or not bool(r.claims_json_valid)))
    c0, c0_fb, fail = np.array(c0), np.array(c0_fb, float), np.array(fail)
    sv = selective(H, c0)
    assert np.allclose(sv, d.final_symmetric_extreme_average.to_numpy(float))
    systems = {"reported": {"H": H, "SV": sv}, "fallback": {"H": H, "SV": selective(H, c0_fb)}}
    keep = ~fail
    systems["exclude"] = {k: v[keep] for k, v in systems["reported"].items()}
    res = {"counts": {"rows_affected_any": int(fail.sum())}}
    for pol, sysd in systems.items():
        yy, ll = (y[keep], langs[keep]) if pol == "exclude" else (y, langs)
        res[pol] = {"n_rows": int(len(yy)),
                    "systems": {k: {m: round(x, 6) for m, x in metrics(v, yy).items()} for k, v in sysd.items()},
                                        "contrasts": contrasts(ll, yy, sysd, [("SV", "H")], PRIMARY_SEED)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-root", default=None)
    args = ap.parse_args()
    if args.evidence_root:
        set_root(args.evidence_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {"status": "EXPLORATORY_POST_HOC", "added": "2026-10-28",
           "development": development(), "additional": additional()}
    (out / "RESULTS.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:6000])
