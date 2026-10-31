#!/usr/bin/env python3
"""Pre-registered holdout analysis. This is the only file that opens the sealed labels.

Primary 1: Spearman(S0) - Spearman(H)
Primary 2: Spearman(S*) - Spearman(S0), where S* uses the repaired translation if and
           only if the frozen repair experiment's matched contrast reads
           "improvement supported", else the whole-document translation.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

import common as C

FULL = ["spearman", "kendall_tau_b", "pearson", "mae", "rmse", "bias", "extreme_over", "extreme_under"]
DIFF = ["spearman", "mae", "rmse", "bias", "extreme_over", "extreme_under"]


def metrics(s, y, full=True):
    e = s - y
    m = {"spearman": float(spearmanr(s, y)[0]), "mae": float(np.abs(e).mean()),
         "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
         "extreme_over": float((e >= 50).mean() * 100), "extreme_under": float((e <= -50).mean() * 100)}
    if full:
        m.update({"kendall_tau_b": float(kendalltau(s, y)[0]), "pearson": float(pearsonr(s, y)[0])})
    return m


def contrasts(langs, y, systems, pairs):
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    uniq = sorted(set(langs))
    idx = {l: np.where(langs == l)[0] for l in uniq}
    draws = {p: {m: [] for m in DIFF} for p in pairs}
    for _ in range(C.BOOTSTRAP_REPLICATES):
        rows = np.concatenate([idx[l] for l in rng.choice(uniq, len(uniq), replace=True)])
        cache = {k: metrics(systems[k][rows], y[rows], full=False) for k in {x for p in pairs for x in p}}
        for a, b in pairs:
            for m in DIFF:
                draws[(a, b)][m].append(cache[a][m] - cache[b][m])
    out = {}
    for a, b in pairs:
        pa, pb = metrics(systems[a], y, False), metrics(systems[b], y, False)
        out[f"{a} - {b}"] = {m: {"estimate": pa[m] - pb[m],
                                 "ci95": [float(np.percentile(draws[(a, b)][m], 2.5)),
                                          float(np.percentile(draws[(a, b)][m], 97.5))]} for m in DIFF}
    return out


def read(ci, higher_is_better):
    lo, hi = ci
    good, bad = (lo > 0, hi < 0) if higher_is_better else (hi < 0, lo > 0)
    return ("improvement supported (95% CI excludes zero)" if good else
            "worsening supported (95% CI excludes zero)" if bad else "inconclusive (95% CI includes zero)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repair-results", required=True, help="RESULTS.json of the frozen repair experiment")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    freeze = json.loads((out / "RUN_FREEZE.json").read_text())
    if C.sha(C.ROOT / "PROTOCOL.json") != freeze["protocol_sha256"]:
        raise RuntimeError("Protocol changed after freeze")
    if json.loads((out / "COMPLETE.json").read_text())["status"] != "ALL_900_ROWS_COMPLETE":
        raise RuntimeError("Inference not complete")
    rep = json.loads(Path(a.repair_results).read_text())
    if rep.get("protocol_sha256") != C.REPAIR_PROTOCOL_SHA or rep.get("status") != "FINAL":
        raise RuntimeError("Repair result is not the frozen final result")
    use_repair = rep["matched_decision"].startswith("improvement supported")

    recs = {}
    for p in (out / "rows").glob("row_*.json"):
        r = json.loads(p.read_text())
        if r["protocol_sha256"] != freeze["protocol_sha256"]:
            raise RuntimeError("Row from another protocol: " + p.name)
        recs[r["holdout_row"]] = r
    if len(recs) != 900:
        raise RuntimeError("Expected 900 row files")

    # ---- the sealed labels are opened here, once, after all outputs exist
    if C.sha(C.LABELS) != freeze["sealed_labels_sha256"]:
        raise RuntimeError("Sealed labels changed")
    C.dump(out / "LABELS_OPENED.json", {"opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "labels_sha256": freeze["sealed_labels_sha256"],
                                        "repair_matched_decision": rep["matched_decision"],
                                        "S_star_uses_repaired_translation": use_repair})
    with C.LABELS.open(newline="") as f:
        labels = {r["holdout_row"]: float(r["q2_avg"]) * 100 for r in csv.DictReader(f)}

    ids, langs, y, cols = [], [], [], {k: [] for k in
                                       ["H", "S0", "S*_T", "S*_R", "base3_T", "base3_R"]}
    missing_h = []
    for hid in sorted(recs, key=int):
        r = recs[hid]
        h = r["H"]["score"]
        if h is None:
            missing_h.append(hid)
            continue
        ids.append(hid); langs.append(r["lang"]); y.append(labels[hid])
        cols["H"].append(h)
        cols["S0"].append(C.selective(h, r["S0"]["capped"]))
        cols["S*_T"].append(C.selective(h, r["NLI_T"]["trained3"]["capped"]))
        cols["S*_R"].append(C.selective(h, r["NLI_R"]["trained3"]["capped"]))
        cols["base3_T"].append(C.selective(h, r["NLI_T"]["base3"]["capped"]))
        cols["base3_R"].append(C.selective(h, r["NLI_R"]["base3"]["capped"]))
    langs, y = np.array(langs), np.array(y)
    sys_ = {k: np.array(v, dtype=float) for k, v in cols.items()}
    star = "S*_R" if use_repair else "S*_T"
    other = "S*_T" if use_repair else "S*_R"
    base = "base3_R" if use_repair else "base3_T"
    sys_["S*"] = sys_[star]

    primary = contrasts(langs, y, sys_, [("S0", "H"), ("S*", "S0")])
    secondary = contrasts(langs, y, sys_, [("S*", "H"), (other, "S0"), ("S*", base), ("S*_R", "S*_T")])
    res = {
        "status": "FINAL", "n_rows": len(ids), "rows_without_holistic_score": missing_h,
        "S_star_definition": star, "repair_matched_decision": rep["matched_decision"],
        "systems": {k: metrics(v, y) for k, v in sys_.items()},
        "primary": primary,
        "primary_reading": {
            "P1 Spearman S0 - H": read(primary["S0 - H"]["spearman"]["ci95"], True),
            "P2 Spearman S* - S0": read(primary["S* - S0"]["spearman"]["ci95"], True)},
        "secondary": secondary,
    }
    d = Counter()
    for r in recs.values():
        d["H_valid_json"] += int(r["H"]["valid_json"]); d["H_recovered"] += int(r["H"]["recovered"])
        d["T_source_english"] += int(r["T"]["source_check"]); d["T_summary_english"] += int(r["T"]["summary_check"])
        d["T_source_hit_limit"] += int(r["T"]["source_hit_limit"])
        d["claims_total_T"] += len(r["T"]["claims"]); d["zero_claim_rows_T"] += int(not r["T"]["claims"])
        d["R_rows_repaired"] += int(r["R"]["source_repaired"] or r["R"]["summary_repaired"])
        d["R_source_english"] += int(r["R"]["source_check"])
        d["S0_verifier_json_failures"] += sum(not v["valid_json"] for v in r["S0"]["verifications"])
        for lab in r["S0"]["labels"]:
            d["S0_label_" + str(lab).strip().lower()] += 1
    res["diagnostics"] = dict(d)
    C.dump(out / "RESULTS.json", res)

    lines = [f"# Holdout confirmation (n = {len(ids)}); S* = {star}", "",
             "| System | " + " | ".join(FULL) + " |", "|---|" + "---|" * len(FULL)]
    for k in ["H", "S0", "S*", other, base]:
        v = res["systems"][k]
        lines.append(f"| {k} | " + " | ".join(f"{v[m]:.3f}" if m in ("spearman", "kendall_tau_b", "pearson")
                                               else f"{v[m]:.2f}" for m in FULL) + " |")
    lines += ["", "| Contrast | Metric | Estimate | 95% CI |", "|---|---|---|---|"]
    for block in (primary, secondary):
        for name, dd in block.items():
            for m in DIFF:
                lines.append(f"| {name} | {m} | {dd[m]['estimate']:+.3f} | [{dd[m]['ci95'][0]:+.3f}, {dd[m]['ci95'][1]:+.3f}] |")
    lines += ["", *[f"{k}: {v}" for k, v in res["primary_reading"].items()], "",
              "Diagnostics: " + json.dumps(res["diagnostics"])]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
