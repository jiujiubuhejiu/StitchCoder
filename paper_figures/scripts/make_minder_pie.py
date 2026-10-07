"""Generate the category pie chart used in the TMLR paper.

Each wedge is identified by an external callout: a two-segment leader starts
at the midpoint of the relevant outer arc, follows a 45-degree diagonal, then
runs horizontally to its category, count, and percentage.
"""
import json
from pathlib import Path
from _paths import DATA_DIR, OUTPUT_DIR
import numpy as np
import matplotlib.pyplot as plt

OUT = OUTPUT_DIR / "minder_pie.pdf"
DATA = json.loads((DATA_DIR / "minder_categories.json").read_text())
categories = DATA["categories"]
counts = DATA["counts"]
total = sum(counts)
assert total == 136, total

# Okabe--Ito-inspired categorical palette: distinct and color-vision-safe.
colors = ["#0072B2", "#56B4E9", "#009E73", "#999999", "#E69F00"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=300)

wedges, _texts = ax.pie(
    counts,
    labels=None,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(
        edgecolor="white",
        linewidth=0.9,
        joinstyle="round",
        antialiased=True,
    ),
)

# The manual horizontal endpoints keep labels compact while avoiding overlap.
# The final slice is routed rightward so its small top wedge remains legible.
callout_sides = [1, -1, -1, -1, 1]
# The two bottom wedges use a downward (-45 degree) diagonal; the remaining
# callouts rise at +45 degrees. This balances the label layout around the pie.
callout_verticals = [-1, -1, 1, 1, 1]
callout_end_x = [1.28, -1.28, -1.32, -0.94, 0.48]
# Radius 1 is the pie's outer arc, so this is the midpoint of each wedge's
# curved boundary rather than a point inside the sector.
leader_radius = 1.0
# The elbow is pushed beyond the unit-radius pie so the change from diagonal
# to horizontal is visibly outside the slice, rather than over its fill.
elbow_radius = 1.18
minimum_diagonal = 0.18

for wedge, category, count, color, side, vertical, end_x in zip(
    wedges,
    categories,
    counts,
    colors,
    callout_sides,
    callout_verticals,
    callout_end_x,
):
    mid_angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
    # Start at the midpoint of the wedge's outer arc. Equal horizontal and
    # vertical offsets make the first segment a +/-45-degree diagonal.
    anchor = (
        leader_radius * np.cos(mid_angle),
        leader_radius * np.sin(mid_angle),
    )
    # Solve ||anchor + d * (side, vertical)|| = elbow_radius.
    linear_term = 2.0 * (side * anchor[0] + vertical * anchor[1])
    constant_term = anchor[0] ** 2 + anchor[1] ** 2 - elbow_radius ** 2
    diagonal_length = max(
        minimum_diagonal,
        (-linear_term + np.sqrt(linear_term ** 2 - 8.0 * constant_term)) / 4.0,
    )
    elbow = (
        anchor[0] + side * diagonal_length,
        anchor[1] + vertical * diagonal_length,
    )
    label = f"{category}\n{count} features ({100.0 * count / total:.1f}%)"

    ax.plot(
        [anchor[0], elbow[0], end_x],
        [anchor[1], elbow[1], elbow[1]],
        color=color,
        linewidth=1.25,
        solid_capstyle="round",
        zorder=3,
    )
    ax.text(
        end_x + side * 0.04,
        elbow[1],
        label,
        ha="left" if side > 0 else "right",
        va="center",
        fontsize=9.5,
        color="#222222",
        linespacing=1.25,
    )

ax.set_axis_off()
ax.set_aspect("equal")
fig.subplots_adjust(bottom=0.03, top=0.97, left=0.03, right=0.97)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print(f"wrote {OUT}")
