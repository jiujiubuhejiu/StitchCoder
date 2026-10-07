"""Render the TMLR DE-scan figure from the frozen 31-point experiment output.

Input:
  outputs/case_study_bias_shift_full_750M_n1000/de_scan_bias_shift_full.json
Outputs:
  outputs/paper/TMLR/TMLR/figures/de_scan.{png,pdf}
  outputs/paper/TMLR/TMLR/figures/de_scan_a.{png,pdf}
  outputs/paper/TMLR/TMLR/figures/de_scan_b.{png,pdf}

The manuscript caption carries the selected-cutoff interpretation; small
unfilled rings identify the corresponding points without adding plot text.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from _paths import DATA_DIR, OUTPUT_DIR

SOURCE = DATA_DIR / "de_scan_bias_shift_full.json"
OUT_DIR = OUTPUT_DIR
SELECTED_K = 1000
P_THRESHOLD = 0.01

# Original scan palette.
BLUE = "#1f77b4"
TEAL = "#2ca02c"
SELECTED = "#d62728"
MUTED = "gray"


def load_scan(path: Path):
    rows = sorted(json.loads(path.read_text()), key=lambda row: row["percentile"])
    percentile = np.array([row["percentile"] for row in rows])
    enrichment = np.array([row["de"] for row in rows])
    p_value = np.array([row["mann_whitney_p"] for row in rows])
    top_k = np.array([row["top_k"] for row in rows])
    return percentile, enrichment, p_value, top_k


def configure_axis(ax):
    ax.set_xscale("log")
    ax.set_xlabel("Candidate-set percentile (\\%)")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def draw_de(ax_de, percentile, enrichment, selected_idx):
    selected_percentile = percentile[selected_idx]
    ax_de.plot(
        percentile,
        enrichment,
        color=BLUE,
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor=BLUE,
        zorder=3,
    )
    ax_de.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax_de.scatter(
        [selected_percentile],
        [enrichment[selected_idx]],
        s=55,
        facecolors="none",
        edgecolors=SELECTED,
        linewidths=1.6,
        zorder=5,
    )
    ax_de.set_ylabel("Differential Enrichment (DE)")


def draw_significance(ax_p, percentile, neg_log_p, selected_idx):
    selected_percentile = percentile[selected_idx]
    ax_p.plot(
        percentile,
        neg_log_p,
        color=TEAL,
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor=TEAL,
        zorder=3,
    )
    ax_p.axhline(-np.log10(P_THRESHOLD), color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax_p.scatter(
        [selected_percentile],
        [neg_log_p[selected_idx]],
        s=55,
        facecolors="none",
        edgecolors=SELECTED,
        linewidths=1.6,
        zorder=5,
    )
    ax_p.set_ylabel(r"$-\log_{10}(p)$")


def save_figure(fig, stem):
    for suffix, dpi in (("pdf", None), ("png", 300)):
        output = OUT_DIR / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(output, **kwargs)


def render():
    percentile, enrichment, p_value, top_k = load_scan(SOURCE)
    selected_idx = int(np.argmin(np.abs(top_k - SELECTED_K)))
    neg_log_p = -np.log10(np.clip(p_value, 1e-300, 1.0))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, (ax_de, ax_p) = plt.subplots(1, 2, figsize=(6.4, 2.6), constrained_layout=True)
    for ax in (ax_de, ax_p):
        configure_axis(ax)
    draw_de(ax_de, percentile, enrichment, selected_idx)
    draw_significance(ax_p, percentile, neg_log_p, selected_idx)
    save_figure(fig, "de_scan")
    plt.close(fig)

    for stem, draw_panel in (
        ("de_scan_a", lambda ax: draw_de(ax, percentile, enrichment, selected_idx)),
        ("de_scan_b", lambda ax: draw_significance(ax, percentile, neg_log_p, selected_idx)),
    ):
        panel_fig, panel_ax = plt.subplots(figsize=(3.2, 2.6), constrained_layout=True)
        configure_axis(panel_ax)
        draw_panel(panel_ax)
        save_figure(panel_fig, stem)
        plt.close(panel_fig)


if __name__ == "__main__":
    render()
