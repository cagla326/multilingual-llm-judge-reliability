"""Descriptive computation summary of the additional run (thesis Section 7.2).

Added 30 October 2026, after all reported results had been inspected. Reads
the archived per-row records only; no model is run.

Counts, per example, the generation calls of the holistic judgement (H) and of
the claim-checking path used by SV (source translation, summary translation,
claim extraction, one verification per claim). Generated text is measured in
characters as a proxy for output tokens, because the run records store the
outputs but not their token counts. Wall time is the per-row ``seconds`` field;
it covers every condition computed for a row in this run (both verifiers, the
untrained control and the repaired translations), so it is an upper bound for
the SV path and cannot be split by step.

Usage: python supplementary_controls/computation_cost_summary.py --out <new dir>
           [--evidence-root <archive root>]
"""
from pathlib import Path
import argparse, collections, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-root", default=None)
    a = ap.parse_args()
    root = Path(a.evidence_root).resolve() if a.evidence_root else Path(__file__).resolve().parents[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(p.read_text()) for p in sorted((root / "archival/additional_run/run_v1/rows").glob("row_*.json"))]
    assert len(rows) == 900

    claims = np.array([len(r["T"]["claims"]) for r in rows])
    assert all(len(r["S0"]["verifications"]) == len(r["T"]["claims"]) for r in rows)
    chars = {
        "H": sum(len(r["H"]["raw_response"]) for r in rows),
        "source_translation": sum(len(r["T"]["source_en"]) for r in rows),
        "summary_translation": sum(len(r["T"]["summary_en"]) for r in rows),
        "claim_extraction": sum(len(r["T"]["raw_claim_extraction"]) for r in rows),
        "verification": sum(len(v["raw_response"]) for r in rows for v in r["S0"]["verifications"]),
    }
    sv_total = sum(chars.values())
    seconds = np.array([r["seconds"] for r in rows], dtype=float)
    calls_sv = 1 + 2 + 1 + claims  # holistic + two translations + extraction + one verification per claim

    res = {
        "status": "DESCRIPTIVE_POST_HOC", "added": "2026-10-30", "n": len(rows),
        "claims": {"total": int(claims.sum()), "mean_per_example": float(claims.mean()),
                   "examples_without_claims": int((claims == 0).sum())},
        "generation_calls_per_example": {"H": 1, "SV_mean": float(calls_sv.mean())},
        "trained_verifier_note": "SV-NLI scores each claim with one forward pass over three label tokens (restricted log-probabilities), not a generated response.",
        "generated_characters": {k: int(v) for k, v in chars.items()},
        "generated_characters_share_pct": {k: round(100 * v / sv_total, 1) for k, v in chars.items()},
        "sv_path_to_H_character_ratio": sv_total / chars["H"],
        "wall_time": {"total_gpu_hours": float(seconds.sum() / 3600), "median_seconds_per_example": float(np.median(seconds)),
                      "mean_seconds_per_example": float(seconds.mean()),
                      "gpus": dict(collections.Counter(r["runtime"]["gpu"] for r in rows)),
                      "scope": "all conditions computed for a row in the additional run; not separable by step"},
    }
    (out / "RESULTS.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
