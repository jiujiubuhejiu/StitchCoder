"""Render the TMLR alignment hierarchy as title-free, captioned panels.

Source data are the fixed d=0.5 hierarchy results in
paper/figures/alignment_ladder_data.json.  The script writes a combined
reference figure and the two panel assets used by LaTeX subcaptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from _paths import DATA_DIR, OUTPUT_DIR

HERE = OUTPUT_DIR
DATA = json.loads((DATA_DIR / "alignment_ladder_data.json").read_text())
SELF = json.loads((DATA_DIR / "alignment_self.json").read_text())
SELF_BSF = SELF["bsf"]
SELF_BSR = SELF["bsr"]
Y_LO, Y_HI = 0.2, 1.12

CELL_ORDER = [
    "Can.→Mat.", "Mat.→Can.",
    "Base→Inst.", "Inst.→Base",
    "2B→9B", "9B→2B",
]
GROUPS = [
    ("cross-SAE", ["Can.→Mat.", "Mat.→Can."]),
    ("base/instruct", ["Base→Inst.", "Inst.→Base"]),
    ("cross-scale", ["2B→9B", "9B→2B"]),
]
RECORDS = {row["cell"]: row for row in DATA}

# Same explicit font family for both panels. The legend size is 25.6% larger
# than the source figure's 7.8 pt; value labels are enlarged proportionally.
FONT = "DejaVu Sans"
LEGEND_SIZE = 9.8
VALUE_SIZE_A = 9.8
VALUE_SIZE_B = 8.9
TICK_SIZE = 10.0
LABEL_SIZE = 11.0
BSF_VALUE_X_OFFSET = -0.035

C_BSF = "#08519C"
C_BSR = "#6BAED6"
C_BSF_DARK = "#062F5C"
C_BSR_DARK = "#2A6BA0"
C_CEIL = "#888888"


def style_axis(ax) -> None:
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"$F_1$ ($d=0.5$)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_aggregate(ax) -> None:
    """Panel (a): category means, with a readable self/ceiling tick label."""
    bar_w = 0.26
    axis_spacing = 0.81
    axis_names = ["self\n(ceiling)"] + [name for name, _ in GROUPS]

    bsf = [SELF_BSF]
    bsr = [SELF_BSR]
    for _, members in GROUPS:
        bsf.append(np.mean([RECORDS[m]["bsf_full"] for m in members]))
        bsr.append(np.mean([RECORDS[m]["bsr_f1"] for m in members]))
    bsf = np.asarray(bsf)
    bsr = np.asarray(bsr)

    centers = np.arange(len(axis_names)) * axis_spacing
    x_bsf = centers - bar_w / 2
    x_bsr = centers + bar_w / 2
    ax.bar(x_bsf, bsf, width=bar_w, color=C_BSF, alpha=0.95,
           edgecolor="white", linewidth=0.7, zorder=2, label="BS-F (axis mean)")
    ax.bar(x_bsr, bsr, width=bar_w, color=C_BSR, alpha=0.95,
           edgecolor="white", linewidth=0.7, zorder=2, label="BS-R (axis mean)")
    ax.plot(x_bsf, bsf, "-o", color=C_BSF_DARK, lw=1.7, ms=5.5,
            markerfacecolor="white", markeredgewidth=1.5, zorder=4)
    ax.plot(x_bsr, bsr, "-o", color=C_BSR_DARK, lw=1.7, ms=5.5,
            markerfacecolor="white", markeredgewidth=1.5, zorder=4)

    for i, (x, value) in enumerate(zip(x_bsf, bsf)):
        if i:
            ax.text(x, value - 0.030, f"{value:.2f}", ha="center", va="top",
                    color="white", fontsize=VALUE_SIZE_A, fontweight="bold")
    for i, (x, value) in enumerate(zip(x_bsr, bsr)):
        if i:
            ax.text(x, value + 0.018, f"{value:.2f}", ha="center", va="bottom",
                    color=C_BSR_DARK, fontsize=VALUE_SIZE_A, fontweight="bold")
    ax.text(centers[0], max(bsf[0], bsr[0]) + 0.020,
            f"{(bsf[0] + bsr[0]) / 2:.2f}", ha="center", va="bottom",
            color="#333", fontsize=VALUE_SIZE_A, fontweight="bold")

    ax.set_xticks(centers)
    ax.set_xticklabels(axis_names, fontsize=LABEL_SIZE)
    for i, label in enumerate(ax.get_xticklabels()):
        if i == 0:
            label.set_rotation(0)
            label.set_ha("center")
        else:
            label.set_rotation(12)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
    ax.set_xlim(-axis_spacing / 2 - 0.05, centers[-1] + axis_spacing / 2 + 0.05)
    style_axis(ax)

    axis_transform = ax.get_xaxis_transform()
    arrow_y = -0.24
    ax.annotate("", xy=(centers[-1], arrow_y), xytext=(centers[0], arrow_y),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
    ax.text((centers[0] + centers[-1]) / 2, arrow_y - 0.035,
            "increasing model difference", ha="center", va="top",
            fontsize=10.5, color="#333", style="italic", transform=axis_transform)
    ax.legend(loc="upper right", frameon=True, fontsize=LEGEND_SIZE,
              handlelength=1.2, handletextpad=0.5, borderpad=0.4,
              framealpha=0.92, edgecolor="#cccccc")


def draw_directional(ax) -> None:
    """Panel (b): direction-specific values and the self-alignment ceiling."""
    bar_w = 0.26
    intra_pair = 0.75
    inter_extra = 0.15
    x_centers = [0.0]
    for i in range(1, len(CELL_ORDER)):
        x_centers.append(x_centers[-1] + intra_pair + (inter_extra if i % 2 == 0 else 0.0))
    x_centers = np.asarray(x_centers)
    x_bsf = x_centers - bar_w / 2
    x_bsr = x_centers + bar_w / 2
    cells = [RECORDS[name] for name in CELL_ORDER]
    bsf = np.asarray([cell["bsf_full"] for cell in cells])
    bsr = np.asarray([cell["bsr_f1"] for cell in cells])

    ax.bar(x_bsf, bsf, width=bar_w, color=C_BSF, alpha=0.97,
           edgecolor="white", linewidth=0.7, zorder=2, label="BS-F (one-to-one)")
    ax.bar(x_bsr, bsr, width=bar_w, color=C_BSR, alpha=0.97,
           edgecolor="white", linewidth=0.7, zorder=2, label="BS-R (one-to-many)")
    for x, value in zip(x_bsf, bsf):
        ax.text(x + BSF_VALUE_X_OFFSET, value + 0.012, f"{value:.2f}", ha="center", va="bottom",
                color=C_BSF, fontsize=VALUE_SIZE_B, fontweight="bold")
    for x, value in zip(x_bsr, bsr):
        ax.text(x, value + 0.012, f"{value:.2f}", ha="center", va="bottom",
                color=C_BSR, fontsize=VALUE_SIZE_B, fontweight="bold")

    ceiling = (SELF_BSF + SELF_BSR) / 2
    x_lo = x_centers[0] - intra_pair / 2 - 0.05
    x_hi = x_centers[-1] + intra_pair / 2 + 0.05
    ax.axhline(ceiling, color=C_CEIL, lw=1.0, linestyle="--", zorder=1)
    ax.text(x_lo + 0.05, ceiling + 0.013, f"self-alignment ceiling $F_1\\approx{ceiling:.3f}$",
            ha="left", va="bottom", fontsize=9.6, color="#666", style="italic")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(CELL_ORDER, rotation=12, fontsize=LABEL_SIZE,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(x_lo, x_hi)
    style_axis(ax)
    label_y = -0.17
    group_centers = [(x_centers[2 * i] + x_centers[2 * i + 1]) / 2 for i in range(len(GROUPS))]
    for center, (name, _) in zip(group_centers, GROUPS):
        ax.text(center, label_y, name, ha="center", va="top", fontsize=10.8,
                color="#333", transform=ax.get_xaxis_transform())
    ax.legend(loc="upper right", frameon=True, fontsize=LEGEND_SIZE,
              handlelength=1.2, handletextpad=0.5, borderpad=0.4,
              framealpha=0.92, edgecolor="#cccccc")


def save_panel(draw, output: Path, *, figsize: tuple[float, float], bottom: float) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.98, bottom=bottom)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    save_panel(draw_aggregate, HERE / "alignment_ladder2_a.pdf", figsize=(7.4, 4.8), bottom=0.31)
    save_panel(draw_directional, HERE / "alignment_ladder2_b.pdf", figsize=(7.4, 4.6), bottom=0.25)

    # Keep an optimized combined asset at the original filename as well.
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.4, 9.0))
    draw_aggregate(ax_a)
    draw_directional(ax_b)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.99, bottom=0.10, hspace=0.78)
    fig.savefig(HERE / "alignment_ladder2.pdf", bbox_inches="tight")
    fig.savefig(HERE / "alignment_ladder2.png", bbox_inches="tight", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
