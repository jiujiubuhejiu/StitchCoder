"""Reproduce the active TMLR causal forest plot using frozen plot rows.

See data/causal_optionC.json for the original approximate-whisker convention.
"""
from __future__ import annotations
import json
from pathlib import Path
from _paths import DATA_DIR, OUTPUT_DIR

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = OUTPUT_DIR
TMLR_OPTION_C_PNG = None

# Restrained, print-friendly palette: slate blue, brick red, and cool grey.
C_INSTR = "#3B6EA5"
C_BASE  = "#C45A54"
C_NEUT  = "#7D8790"
C_DEGEN = "#B58936"
C_REF   = "#BBBBBB"

# ColorBrewer-inspired blue--red--grey tints: distinct in lightness, muted
# enough for a paper background, and paired with the darker foreground marks.
GROUP_BAND_COLOR = {
    "null": "#E9ECEF",
    "pc_clean": "#DCEAF6",
    "pc_degen": "#F9DCD9",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

def plot_option_c(*, band_colors=GROUP_BAND_COLOR,
                  output_stem="causal_optionC",
                  tmlr_png=TMLR_OPTION_C_PNG):
    # Each row: (label, delta, ci_half, group, degen_flag)
    # CI half-width approximated as 0.12 per the paper's reported bound.
    rows = json.loads((DATA_DIR / "causal_optionC.json").read_text())["rows"]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y_positions = np.arange(n)[::-1]  # top → bottom

    group_color = {
        "null":     C_NEUT,
        "pc_clean": C_INSTR,
        "pc_degen": C_BASE,
    }

    for y, (lab, d, ci, grp, degen_flag) in zip(y_positions, rows):
        color = group_color[grp]
        ax.errorbar(d, y, xerr=ci, fmt="o",
                    color=color, ecolor=color,
                    ms=5.5, capsize=2.5, lw=1.2,
                    mfc=color if not degen_flag else "white",
                    mec=color, mew=1.2)

    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    # Color tick labels by group, so the grouping is readable without
    # any rotated text on the right.
    for tick, (_, _, _, grp, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(group_color[grp])
    ax.set_xlim(-1.10, 0.50)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_xlabel(r"$\Delta$ Refusal Rate")
    ax.grid(axis="x", color="0.86", linewidth=0.55, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Group dividers — pale background bands only, no rotated labels
    # (group identity is conveyed by the colored y-tick labels).
    group_bands = [
        ("null",     [0, 1, 2]),
        ("pc_clean", [3, 4, 5, 6, 7]),
        ("pc_degen", [8, 9, 10, 11, 12]),
    ]
    for grp, idxs in group_bands:
        ys = [y_positions[i] for i in idxs]
        ymin, ymax = min(ys) - 0.45, max(ys) + 0.45
        ax.axhspan(ymin, ymax, color=band_colors[grp], zorder=0)
    # Match the categorical axis to the outer group bands: no accidental
    # white margins above the grey block or below the red block.
    ax.set_ylim(-0.45, n - 0.55)

    # Compact key for the three intervention families and hollow markers.
    legend_handles = [
        mpatches.Patch(color=group_color["null"],
                       label="Candidate-feature tests"),
        mpatches.Patch(color=group_color["pc_clean"],
                       label="Suppression control (Instruct)"),
        mpatches.Patch(color=group_color["pc_degen"],
                       label="Induction control (Base)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.5,
                   markerfacecolor="white", markeredgecolor=C_DEGEN,
                   markeredgewidth=1.2, label="Generation collapse"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              frameon=False, handlelength=1.0, handleheight=0.7)

    fig.tight_layout()
    out = OUT / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if tmlr_png is not None:
        fig.savefig(tmlr_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    plot_option_c()
