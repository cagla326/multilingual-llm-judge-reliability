"""Exploratory descriptive check: where does the holistic over-scoring sit?

Added 28 October 2026, after all reported results had been inspected. No model
is run and no reported score changes.

The released Q2 human scores are averages of binary attribution votes, so
almost all of them take the values 0, 1/3, 2/3 or 1. The judge instead chooses
graded anchors (0, 25, 50, 75, 100), where 75 means "mostly supported, with
minor unsupported or imprecise details". This script tabulates, for each
judge score, how many rows receive it, their mean human score, and the share
of the total signed error (bias) that those rows contribute, on the
development and the additional sample.

Usage:  python supplementary_controls/scale_mismatch_check.py --out <new dir>
        [--evidence-root <extracted evidence archive>]   (default: this archive)
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(H, y):
    e = H - y
    vote_like = np.isclose(y[:, None], [0, 100 / 3, 200 / 3, 100], atol=0.05).any(axis=1)
    out = {"n": int(len(y)), "bias": round(float(e.mean()), 2),
           "human_scores_at_vote_shares_pct": round(float(vote_like.mean() * 100), 1), "by_judge_score": {}}
    for h in sorted(np.unique(H)):
        m = H == h
        out["by_judge_score"][str(h)] = {
            "rows": int(m.sum()), "mean_human": round(float(y[m].mean()), 1),
            "mean_signed_error": round(float(e[m].mean()), 1),
            "share_of_total_signed_error_pct": round(float(e[m].sum() / e.sum() * 100), 1)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-root", default=None)
    args = ap.parse_args()
    if args.evidence_root:
        ROOT = Path(args.evidence_root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = pd.read_csv(ROOT / "supplementary_controls/data/development900.csv")
    add = pd.read_csv(ROOT / "supplementary_controls/results_v1/additional_scores.csv")
    res = {"status": "EXPLORATORY_POST_HOC", "added": "2026-10-28",
           "development": table(dev.gemma_fullsource_score.to_numpy(float), dev.human_0_100.to_numpy(float)),
           "additional": table(add.H.to_numpy(float), add.y.to_numpy(float))}
    (out / "RESULTS.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
