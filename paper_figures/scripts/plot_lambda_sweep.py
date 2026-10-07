"""BS-R ridge-λ sensitivity figure for Appendix B.

Reads the per-λ remix_f1.json payloads written by
run_remix_lambda_sensitivity.py and renders a publication-targeted F1/P/R
curve over log-λ. The paper's default λ=100 is highlighted (not "best λ"),
so the figure answers the reviewer's question directly: how does the
reported BS-R F1 move when λ changes by ±1 decade.
"""
from __future__ import annotations
import json
from pathlib import Path
from _paths import DATA_DIR, OUTPUT_DIR

import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = DATA_DIR / "lambda_sweep"

# Same palette family as the alignment-ladder figure for visual consistency.
C_F1 = "#4C72B0"   # steel blue
C_P  = "#55A868"   # sage green
C_R  = "#C44E52"   # muted red
C_REF = "#888888"

LAMBDAS = [10, 30, 100, 300, 1000]

rows = []
for lam in LAMBDAS:
    path = SWEEP_DIR / f"lambda_{lam}.json"
    payload = json.loads(path.read_text())
    m = payload["f1_remix"]
    rows.append({"lam": float(lam),
                 "F1": m["f1"], "P": m["precision"], "R": m["recall"]})

lams = np.array([r["lam"] for r in rows])
f1s  = np.array([r["F1"]  for r in rows])
ps   = np.array([r["P"]   for r in rows])
rs   = np.array([r["R"]   for r in rows])

fig, ax = plt.subplots(figsize=(5.6, 3.6))

ax.plot(lams, ps, marker="s", linestyle="--", color=C_P,  lw=1.4, ms=6,
        markerfacecolor="white", markeredgewidth=1.6, alpha=0.85, label="Precision")
ax.plot(lams, rs, marker="^", linestyle="--", color=C_R,  lw=1.4, ms=6,
        markerfacecolor="white", markeredgewidth=1.6, alpha=0.85, label="Recall")
ax.plot(lams, f1s, marker="o", color=C_F1, lw=2.0, ms=7,
        markerfacecolor="white", markeredgewidth=1.8, label="F1 (BS-R)", zorder=5)

# Highlight the paper's choice (λ=100), not the empirical best — the reviewer's
# question is about the chosen value, not about retuning.
i100 = list(LAMBDAS).index(100)
ax.axvline(100, color=C_REF, linestyle=":", lw=1.2, alpha=0.8)
ax.annotate(f"paper default\n$\\lambda=100$\nF1 = {f1s[i100]:.3f}",
            xy=(100, f1s[i100]),
            xytext=(150, f1s[i100] - 0.022),
            fontsize=8.6, color="#333",
            ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color=C_REF, lw=0.8, alpha=0.8))

# Plateau annotation: λ=100 is within +0.010 of the empirical max at λ=1000.
plateau_lo, plateau_hi = f1s[i100], f1s[-1]
ax.annotate("",
            xy=(1000, plateau_hi),  xytext=(100, plateau_lo),
            arrowprops=dict(arrowstyle="<->", color="#666", lw=0.9, alpha=0.6))
ax.text(316, (plateau_lo + plateau_hi) / 2 + 0.004,
        rf"$\Delta\mathrm{{F1}}={plateau_hi-plateau_lo:+.3f}$",
        fontsize=8.2, color="#444", ha="center", style="italic")

ax.set_xscale("log")
ax.set_xticks(LAMBDAS)
ax.set_xticklabels([str(l) for l in LAMBDAS])
ax.set_xlabel(r"Ridge penalty $\lambda$ (log scale)", fontsize=10)
ax.set_ylabel("Score on held-out tokens", fontsize=10)
ax.set_ylim(0.59, 0.71)
ax.grid(True, axis="y", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.legend(loc="lower right", frameon=True, fontsize=9.0,
          handlelength=1.6, framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
out_pdf = OUTPUT_DIR / "lambda_sweep.pdf"
out_png = OUTPUT_DIR / "lambda_sweep.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
