"""Result figures added on 29 October 2026 (thesis Figures in Chapters 4 and 7).

Reads saved records only; no model is run. Every plotted value is either printed
in the thesis tables or recomputed here from the archived development rows.

  agreement_by_language.pdf  per-language Spearman correlation of the initial
                             Q2 judgement on the development sample, grouped
                             by resource level (Section 4.2)
  overscoring_by_score.pdf   mean human score for each judge score on the
                             development sample (Section 4.3)
  level_vs_order.pdf         MAE against Spearman for the systems in
                             Table 19 on the additional sample (Section 7.4)

Usage: python figures/make_result_figures.py --out <dir> [--evidence-root <archive root>]
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f94", "#1f2933", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)


def agreement_by_language(root, out):
    d = pd.read_csv(root / "supplementary_controls/data/development900.csv")
    res = pd.read_csv(root / "archival/development/language_resource_levels_from_xlsum_counts.csv")
    lang_col = [c for c in res.columns if "lang" in c.lower()][0]
    lvl_col = [c for c in res.columns if "level" in c.lower() or "group" in c.lower()][0]
    level = dict(zip(res[lang_col].str.lower(), res[lvl_col].str.lower()))
    rows = []
    for lang, g in d.groupby("lang"):
        rows.append((lang, level[lang.lower()], spearmanr(g.gemma_fullsource_score, g.human_0_100)[0]))
    t = pd.DataFrame(rows, columns=["lang", "level", "rho"])
    order = [("high", "High", BLUE), ("medium", "Medium", ORANGE), ("low", "Low", AQUA)]
    markers = {"high": "o", "medium": "s", "low": "^"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rng = np.random.default_rng(0)
    for x, (key, label, col) in enumerate(order):
        v = t[t.level == key].rho.dropna().to_numpy()
        jit = rng.uniform(-.16, .16, len(v))
        ax.scatter(x + jit, v, s=34, color=col, marker=markers[key], edgecolor="white", linewidth=.8, zorder=3)
        ax.plot([x - .28, x + .28], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    ax.axhline(0, color=MUTED, lw=.8, ls="--")
    ax.set_xticks(range(3), [f"{lab}\n({(t.level == k).sum()} languages)" for k, lab, _ in order])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("Spearman correlation\nwith human Q2 ratings")
    style(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out / "agreement_by_language.pdf", bbox_inches="tight"); plt.close(fig)
    return t


def overscoring_by_score(root, out):
    s = json.loads((root / "supplementary_controls/results_scale_v1/RESULTS.json").read_text())["development"]
    anchors = [0.0, 25.0, 50.0, 75.0, 100.0]
    b = s["by_judge_score"]
    x = anchors; y = [b[str(a)]["mean_human"] for a in anchors]; n = [b[str(a)]["rows"] for a in anchors]
    share = [b[str(a)]["share_of_total_signed_error_pct"] for a in anchors]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1)
    ax.text(2, 96, "Dashed line: judge score = human score", ha="left", fontsize=8.5, color=MUTED)
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, xi], color=ORANGE if xi > yi else BLUE, lw=2, alpha=.7, zorder=2)
    ax.scatter(x, y, s=[max(30, k * 1.1) for k in n], color=INK, edgecolor="white", linewidth=1, zorder=3)
    for xi, yi, k, sh in zip(x, y, n, share):
        shs = f"{sh:.1f}" if abs(sh) < 2 else f"{sh:.0f}"
        ax.annotate(f"n = {k}\n{shs}% of bias".replace("-", "\u2212"), (xi, yi), xytext=(16 if xi == 0 else 10, -22),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xticks(anchors); ax.set_xlim(-8, 112); ax.set_ylim(-5, 108)
    ax.set_xlabel("Judge's holistic Q2 score")
    ax.set_ylabel("Mean human score\n(rescaled votes)")
    style(ax)
    fig.tight_layout(); fig.savefig(out / "overscoring_by_score.pdf", bbox_inches="tight"); plt.close(fig)


def level_vs_order(out):
    # Values as printed in Table 19 (additional sample).
    pts = [("H", 33.41, .275, GREY, "o"), ("CalH", 29.71, .275, GREY, "D"),
           ("Mean", 31.69, .338, BLUE, "o"), ("SV", 31.35, .321, BLUE, "s"),
           ("Mean-NLI", 30.36, .394, ORANGE, "o"), ("SV-NLI", 29.64, .390, ORANGE, "s"),
           ("Cal. Mean-NLI", 28.42, .394, ORANGE, "D"), ("Cal. SV-NLI", 27.61, .390, ORANGE, "D")]
    offs = {"H": (-4, 7), "CalH": (6, -12), "Mean": (6, 4), "SV": (6, -11), "Mean-NLI": (6, 5),
            "SV-NLI": (-4, -14), "Cal. Mean-NLI": (-22, 9), "Cal. SV-NLI": (-10, -14)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for lab, mae, rho, col, mk in pts:
        ax.scatter([mae], [rho], s=60, color=col, marker=mk, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(lab, (mae, rho), xytext=offs[lab], textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate("", xy=(29.9, .275), xytext=(33.15, .275),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.text(31.55, .281, "offset: lower error,\nsame order", ha="center", fontsize=8, color=MUTED)
    ax.set_xlim(34.2, 26.9)
    ax.set_ylim(.25, .42)
    ax.set_xlabel("MAE in score points (lower is better; axis reversed)")
    ax.set_ylabel("Spearman correlation\n(higher is better)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=GREY, marker="o", ls="", label="Judge alone"),
         Line2D([], [], color=BLUE, marker="o", ls="", label="With original verifier"),
         Line2D([], [], color=ORANGE, marker="o", ls="", label="With trained verifier"),
         Line2D([], [], color=MUTED, marker="s", ls="", label="Selective rule"),
         Line2D([], [], color=MUTED, marker="D", ls="", label="Offset-calibrated")]
    ax.legend(handles=h, loc="upper left", fontsize=8, frameon=False, ncol=1)
    style(ax)
    fig.tight_layout(); fig.savefig(out / "level_vs_order.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-root", default=None)
    a = ap.parse_args()
    root = Path(a.evidence_root).resolve() if a.evidence_root else Path(__file__).resolve().parents[1]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = agreement_by_language(root, out)
    overscoring_by_score(root, out)
    level_vs_order(out)
    print(t.groupby("level").rho.describe().round(3).to_string())
